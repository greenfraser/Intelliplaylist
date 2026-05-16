from __future__ import annotations

import base64
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlencode


import requests
from flask import Flask, jsonify, redirect, request, session
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import (
    has_track_column,
    has_artist_match,
    has_album_match,
    has_track_name_match,
)
from src.nlp.openai_llm_parser import openai_llm_callable
from src.nlp.request_parser import parse_playlist_request
from src.nlp.request_schema import PlaylistRequest
from src.asp.playlist import build_playlist_with_metadata

DEFAULT_MIN_POPULARITY = 70
SPOTIFY_ACCOUNT_BASE = "https://accounts.spotify.com"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_SCOPES = "playlist-modify-private playlist-modify-public"
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://127.0.0.1:5173")
SPOTIFY_REDIRECT_URI = os.environ.get(
    "SPOTIFY_REDIRECT_URI",
    "http://127.0.0.1:5001/api/spotify/callback",
)
SPOTIFY_OAUTH_STATES: set[str] = set()


def _has_specific_filters(request_obj: PlaylistRequest) -> bool:
    return bool(
        request_obj.include_genres
        or request_obj.exclude_genres
        or request_obj.year is not None
        or request_obj.year_range is not None
        or request_obj.decade is not None
        or request_obj.include_artists
        or request_obj.exclude_artists
        or request_obj.include_albums
        or request_obj.exclude_albums
        or request_obj.include_tracks
        or request_obj.exclude_tracks
        or request_obj.duration_range_ms is not None
        or request_obj.tempo_range is not None
        or request_obj.energy_range is not None
        or request_obj.valence_range is not None
        or request_obj.danceability_range is not None
        or request_obj.acousticness_range is not None
        or request_obj.instrumentalness_range is not None
        or request_obj.liveness_range is not None
        or request_obj.speechiness_range is not None
        or request_obj.exclude_explicit
        or request_obj.exclude_live
        or request_obj.exclude_remix
        or request_obj.exclude_christmas
    )

def strip_unsupported_database_constraints(request_obj: PlaylistRequest) -> PlaylistRequest:
    if not has_track_column("year"):
        request_obj.year = None
        request_obj.year_range = None
        request_obj.decade = None

    return request_obj

def apply_request_defaults(request_obj: PlaylistRequest) -> PlaylistRequest:
    has_required_track = bool(request_obj.include_tracks)

    if request_obj.min_popularity is None and request_obj.max_popularity is None:
        if _has_specific_filters(request_obj):
            request_obj.min_popularity = None
        else:
            request_obj.min_popularity = DEFAULT_MIN_POPULARITY

    if has_required_track and request_obj.playlist_size is None and request_obj.duration_minutes is None:
        request_obj.playlist_size = 10

    return request_obj




def row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "artists": row[2],
        "tempo": row[3] or 0.0,
        "energy": row[4] or 0.0,
        "valence": row[5] or 0.0,
        "danceability": row[6] or 0.0,
        "acousticness": row[7] or 0.0,
        "instrumentalness": row[8] or 0.0,
        "liveness": row[9] or 0.0,
        "speechiness": row[10] or 0.0,
        "album": row[11] or "",
        "explicit": bool(row[12]),
        "duration_ms": row[13] or 0,
        "year": row[14] or 0,
        "popularity": row[15] or 0,
        "genre": row[16] or "",
    }


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _spotify_credentials_ready() -> bool:
    return bool(_env("SPOTIFY_CLIENT_ID") and _env("SPOTIFY_CLIENT_SECRET"))


def _spotify_basic_auth_header() -> str:
    raw = f"{_env('SPOTIFY_CLIENT_ID')}:{_env('SPOTIFY_CLIENT_SECRET')}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _build_spotify_auth_url() -> str:
    if not _spotify_credentials_ready():
        raise RuntimeError(
            "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
        )

    state = secrets.token_urlsafe(24)

    # Store in both places. The in-memory set is more reliable for local dev.
    session["spotify_oauth_state"] = state
    SPOTIFY_OAUTH_STATES.add(state)

    params = {
        "client_id": _env("SPOTIFY_CLIENT_ID"),
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPES,
        "state": state,
        "show_dialog": "true",
    }

    return f"{SPOTIFY_ACCOUNT_BASE}/authorize?{urlencode(params)}"

def _save_spotify_tokens(token_payload: dict[str, Any]) -> None:
    access_token = str(token_payload.get("access_token") or "")
    refresh_token = str(
        token_payload.get("refresh_token")
        or session.get("spotify_tokens", {}).get("refresh_token")
        or ""
    )
    expires_in = int(token_payload.get("expires_in") or 3600)

    session["spotify_tokens"] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in - 60,
    }


def _exchange_spotify_code_for_tokens(code: str) -> None:
    response = requests.post(
        f"{SPOTIFY_ACCOUNT_BASE}/api/token",
        headers={"Authorization": _spotify_basic_auth_header()},
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Spotify token exchange failed: {response.text}")

    _save_spotify_tokens(response.json())


def _refresh_spotify_access_token() -> Optional[str]:
    tokens = session.get("spotify_tokens") or {}
    refresh_token = str(tokens.get("refresh_token") or "")
    if not refresh_token:
        return None

    response = requests.post(
        f"{SPOTIFY_ACCOUNT_BASE}/api/token",
        headers={"Authorization": _spotify_basic_auth_header()},
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=20,
    )

    if response.status_code != 200:
        session.pop("spotify_tokens", None)
        return None

    _save_spotify_tokens(response.json())
    return str(session["spotify_tokens"].get("access_token") or "")


def _get_spotify_access_token() -> Optional[str]:
    tokens = session.get("spotify_tokens") or {}
    access_token = str(tokens.get("access_token") or "")
    expires_at = float(tokens.get("expires_at") or 0)

    if not access_token:
        return None

    if expires_at <= time.time():
        return _refresh_spotify_access_token()

    return access_token


def _spotify_request(
    method: str,
    path: str,
    *,
    token: str,
    expected_statuses: tuple[int, ...] = (200,),
    **kwargs: Any,
) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    headers.setdefault("Content-Type", "application/json")

    response = requests.request(
        method,
        f"{SPOTIFY_API_BASE}{path}",
        headers=headers,
        timeout=20,
        **kwargs,
    )

    if response.status_code not in expected_statuses:
        raise RuntimeError(f"Spotify API request failed ({response.status_code}): {response.text}")

    if not response.content:
        return {}

    return response.json()


def _first_artist(artists: object) -> str:
    text = str(artists or "").strip()
    if not text:
        return ""

    text = text.strip("[]")
    text = text.replace("'", "").replace('"', "")
    parts = re.split(r";|,|\s+feat\.?\s+|\s+ft\.?\s+", text, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip()

def _format_missing(label: str, values: list[str]) -> str:
    if not values:
        return ""

    quoted = ", ".join(f"“{value}”" for value in values)
    return f"{label}: {quoted}"


def validate_request_against_database(
    request_obj: PlaylistRequest,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    missing_include_artists = [
        artist for artist in request_obj.include_artists
        if not has_artist_match(artist)
    ]
    missing_exclude_artists = [
        artist for artist in request_obj.exclude_artists
        if not has_artist_match(artist)
    ]

    missing_include_albums = [
        album for album in request_obj.include_albums
        if not has_album_match(album)
    ]
    missing_exclude_albums = [
        album for album in request_obj.exclude_albums
        if not has_album_match(album)
    ]

    missing_include_tracks = [
        track for track in request_obj.include_tracks
        if not has_track_name_match(track)
    ]
    missing_exclude_tracks = [
        track for track in request_obj.exclude_tracks
        if not has_track_name_match(track)
    ]

    required_missing_parts = [
        _format_missing("artists not in the dataset", missing_include_artists),
        _format_missing("albums not in the dataset", missing_include_albums),
        _format_missing("tracks not in the dataset", missing_include_tracks),
    ]
    required_missing_parts = [part for part in required_missing_parts if part]

    if required_missing_parts:
        errors.append(
            "I could not generate this playlist because these requested items were not found: "
            + "; ".join(required_missing_parts)
            + "."
        )

    ignored_parts = [
        _format_missing("excluded artists not found", missing_exclude_artists),
        _format_missing("excluded albums not found", missing_exclude_albums),
        _format_missing("excluded tracks not found", missing_exclude_tracks),
    ]
    ignored_parts = [part for part in ignored_parts if part]

    if ignored_parts:
        warnings.append(
            "Some exclusion constraints were ignored because they were not found in the dataset: "
            + "; ".join(ignored_parts)
            + "."
        )

    return errors, warnings


def _search_spotify_track(token: str, track: dict[str, Any]) -> Optional[dict[str, str]]:
    name = str(track.get("name") or "").strip()
    artists = str(track.get("artists") or "").strip()
    first_artist = _first_artist(artists)

    if not name:
        return None

    queries = []
    if first_artist:
        queries.append(f'track:"{name}" artist:"{first_artist}"')
    if artists:
        queries.append(f"{name} {artists}")
    queries.append(name)

    seen_queries = set()
    for query in queries:
        if query in seen_queries:
            continue
        seen_queries.add(query)

        payload = _spotify_request(
            "GET",
            "/search",
            token=token,
            params={"q": query, "type": "track", "limit": 1},
        )
        items = payload.get("tracks", {}).get("items", [])
        if not items:
            continue

        item = items[0]
        spotify_artists = ", ".join(
            artist.get("name", "")
            for artist in item.get("artists", [])
            if artist.get("name")
        )
        return {
            "original_name": name,
            "original_artists": artists,
            "spotify_name": str(item.get("name") or ""),
            "spotify_artists": spotify_artists,
            "spotify_uri": str(item.get("uri") or ""),
            "spotify_url": str(item.get("external_urls", {}).get("spotify") or ""),
        }

    return None


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-intelliplaylist-secret-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

CORS(
    app,
    resources={r"/api/*": {"origins": [FRONTEND_URL, "http://127.0.0.1:5173", "http://localhost:5173"]}},
    supports_credentials=True,
)


@app.get("/api/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.get("/api/spotify/status")
def spotify_status():
    return jsonify({"connected": _get_spotify_access_token() is not None})


@app.get("/api/spotify/login")
def spotify_login():
    try:
        return jsonify({"auth_url": _build_spotify_auth_url()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/spotify/callback")
def spotify_callback():
    error = request.args.get("error")
    if error:
        return redirect(f"{FRONTEND_URL}?spotify=error")

    state = request.args.get("state")
    expected_state = session.get("spotify_oauth_state")

    state_matches_session = bool(state and expected_state and state == expected_state)
    state_matches_memory = bool(state and state in SPOTIFY_OAUTH_STATES)

    if not state_matches_session and not state_matches_memory:
        return redirect(f"{FRONTEND_URL}?spotify=state-error")

    if state:
        SPOTIFY_OAUTH_STATES.discard(state)

    session.pop("spotify_oauth_state", None)
    code = request.args.get("code")
    if not code:
        return redirect(f"{FRONTEND_URL}?spotify=missing-code")

    try:
        _exchange_spotify_code_for_tokens(code)
        return redirect(f"{FRONTEND_URL}?spotify=connected")
    except Exception:
        return redirect(f"{FRONTEND_URL}?spotify=token-error")


@app.post("/api/parse-request")
def parse_request():
    try:
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or "").strip()

        if not text:
            return jsonify({"error": "Missing request text."}), 400

        parsed = parse_playlist_request(
            text,
            llm_callable=openai_llm_callable,
            debug=True,
        )

        parsed = strip_unsupported_database_constraints(parsed)
        parsed = apply_request_defaults(parsed)

        return jsonify({"request": parsed.to_dict()})
    except Exception as exc:
        return jsonify({"error": f"Failed to parse request: {exc}"}), 500


@app.post("/api/generate-playlist")
def generate_playlist():
    print("\n[API] /api/generate-playlist route entered", flush=True)

    try:
        body = request.get_json(silent=True) or {}
        print("[API] JSON body received", flush=True)

        request_data = body.get("request")
        print("[API] request object type:", type(request_data), flush=True)

        if not isinstance(request_data, dict):
            print("[API] ERROR: Missing request object", flush=True)
            return jsonify({"error": "Missing request object."}), 400

        playlist_request = PlaylistRequest.from_dict(request_data)
        print("[API] PlaylistRequest created:", playlist_request.to_dict(), flush=True)
        playlist_request = strip_unsupported_database_constraints(playlist_request)
        playlist_request = apply_request_defaults(playlist_request)
        print("[API] Defaults applied:", playlist_request.to_dict(), flush=True)

        validation_errors, validation_warnings = validate_request_against_database(playlist_request)

        if validation_errors:
            return jsonify({
                "request": playlist_request.to_dict(),
                "tracks": [],
                "generation_note": " ".join(validation_errors),
            })
        
        print("[API] Starting build_playlist_with_metadata...", flush=True)
        result = build_playlist_with_metadata(playlist_request, debug=True)
        print(f"[API] build_playlist_with_metadata finished with {len(result.rows)} rows", flush=True)

        generation_note = result.generation_note

        if validation_warnings:
            warning_text = " ".join(validation_warnings)
            generation_note = f"{warning_text} {generation_note or ''}".strip()

        response_payload = {
            "request": playlist_request.to_dict(),
            "tracks": [row_to_dict(row) for row in result.rows],
            "generation_note": generation_note,
        }

        print("[API] Returning JSON response", flush=True)
        return jsonify(response_payload)

    except Exception as exc:
        print("[API] ERROR in /api/generate-playlist:", repr(exc), flush=True)
        return jsonify({"error": f"Failed to generate playlist: {exc}"}), 500


@app.post("/api/export-spotify")
def export_spotify_playlist():
    try:
        body = request.get_json(silent=True) or {}
        tracks = body.get("tracks")
        playlist_name = str(body.get("playlist_name") or "IntelliPlaylist").strip()

        if not isinstance(tracks, list) or not tracks:
            return jsonify({"error": "No tracks were provided for Spotify export."}), 400

        if not _spotify_credentials_ready():
            return jsonify({
                "error": "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
            }), 500

        token = _get_spotify_access_token()
        if token is None:
            return jsonify({
                "error": "Spotify login required.",
                "auth_url": _build_spotify_auth_url(),
            }), 401

        matched_tracks: list[dict[str, str]] = []
        missing_tracks: list[str] = []
        spotify_uris: list[str] = []
        seen_uris = set()

        for track in tracks:
            if not isinstance(track, dict):
                continue

            match = _search_spotify_track(token, track)
            if not match or not match.get("spotify_uri"):
                missing_tracks.append(
                    f"{track.get('name', 'Unknown track')} by {track.get('artists', 'Unknown artist')}"
                )
                continue

            spotify_uri = match["spotify_uri"]
            if spotify_uri in seen_uris:
                continue

            matched_tracks.append(match)
            spotify_uris.append(spotify_uri)
            seen_uris.add(spotify_uri)

        if not spotify_uris:
            return jsonify({"error": "Spotify could not match any of the generated tracks."}), 404

        playlist = _spotify_request(
            "POST",
            "/me/playlists",
            token=token,
            expected_statuses=(200, 201),
            json={
                "name": playlist_name[:100] or "IntelliPlaylist",
                "public": False,
                "description": "Created by IntelliPlaylist.",
            },
        )

        playlist_id = str(playlist.get("id") or "")
        if not playlist_id:
            return jsonify({"error": "Spotify created a playlist response without an id."}), 500

        for batch in _chunks(spotify_uris, 100):
            _spotify_request(
                "POST",
                f"/playlists/{playlist_id}/items",
                token=token,
                expected_statuses=(200, 201),
                json={"uris": batch},
            )

        return jsonify({
            "playlist_id": playlist_id,
            "playlist_url": playlist.get("external_urls", {}).get("spotify", ""),
            "requested_count": len(tracks),
            "matched_count": len(spotify_uris),
            "missing_tracks": missing_tracks,
            "matched_tracks": matched_tracks,
        })

    except Exception as exc:
        return jsonify({"error": f"Failed to export playlist to Spotify: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
