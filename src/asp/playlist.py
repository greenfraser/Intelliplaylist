from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import ast
import random

from src.data.db import (
    Range,
    TrackRow,
    get_candidates_for_feature_ranges,
    get_tracks_by_ids,
    get_tracks_by_names,
    get_tracks_for_albums,
    get_tracks_for_artists,
)
from src.nlp.emotions import compute_feature_ranges, text_to_emotion_weights
from src.nlp.request_schema import PlaylistRequest
from src.asp.facts import write_rows_to_facts
from src.asp.solver import run_clingo

# Core playlist-building logic for IntelliPlaylist.
# This file turns a parsed PlaylistRequest into database candidates, ASP facts,
# solver output, fallback selections, and a final ordered playlist.

_ACTIVITY_PRESETS: Dict[str, Dict[str, Range]] = {
    "workout": {
        "energy": (0.75, 1.00),
        "tempo": (120.0, 165.0),
        "danceability": (0.50, 1.00),
    },
    "study": {
        "energy": (0.20, 0.50),
        "tempo": (65.0, 115.0),
        "instrumentalness": (0.10, 0.90),
        "speechiness": (0.00, 0.15),
    },
    "party": {
        "energy": (0.70, 1.00),
        "tempo": (115.0, 160.0),
        "danceability": (0.60, 1.00),
    },
    "sleep": {
        "energy": (0.00, 0.25),
        "tempo": (50.0, 95.0),
        "acousticness": (0.30, 1.00),
    },
    "relax": {
        "energy": (0.10, 0.40),
        "tempo": (60.0, 105.0),
        "acousticness": (0.20, 0.90),
    },
}


# Stores the generated playlist rows and any user-facing generation note.
@dataclass
class PlaylistBuildResult:
    rows: List[TrackRow]
    generation_note: Optional[str] = None

# Returns the midpoint of a range, or a default value if no range is provided.
def _midpoint(rng: Optional[Range], default: float) -> float:
    if rng is None:
        return default
    return (rng[0] + rng[1]) / 2.0

# Converts a database row into a dictionary so feature distances can be calculated.
def _row_to_track(row: TrackRow) -> Dict[str, Any]:
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
        "explicit": row[12] or 0,
        "duration_ms": row[13] or 0,
        "year": row[14] or 0,
        "popularity": row[15] or 0,
        "genre": row[16] or "",
        "row": row,
    }


# Builds the target audio-feature profile from the resolved feature ranges.
def _target_profile(ranges: Dict[str, Optional[Range]]) -> Dict[str, float]:
    return {
        "valence": _midpoint(ranges.get("valence"), 0.5),
        "energy": _midpoint(ranges.get("energy"), 0.5),
        "tempo": _midpoint(ranges.get("tempo"), 120.0),
        "danceability": _midpoint(ranges.get("danceability"), 0.5),
        "acousticness": _midpoint(ranges.get("acousticness"), 0.5),
        "instrumentalness": _midpoint(ranges.get("instrumentalness"), 0.5),
        "liveness": _midpoint(ranges.get("liveness"), 0.5),
        "speechiness": _midpoint(ranges.get("speechiness"), 0.5),
    }

# Calculates how closely a track matches the target audio-feature profile.
def _distance_to_target(track: Dict[str, Any], target: Dict[str, float]) -> float:
    score = (
        abs(track["tempo"] - target["tempo"]) / 40.0
        + 1.2 * abs(track["energy"] - target["energy"])
        + 1.0 * abs(track["valence"] - target["valence"])
        + 0.8 * abs(track["danceability"] - target["danceability"])
        + 0.4 * abs(track["acousticness"] - target["acousticness"])
        + 0.3 * abs(track["instrumentalness"] - target["instrumentalness"])
        + 0.2 * abs(track["liveness"] - target["liveness"])
        + 0.2 * abs(track["speechiness"] - target["speechiness"])
    )
    return score

# Calculates the transition cost between two neighbouring tracks.
def _transition_cost(
    prev_track: Dict[str, Any],
    next_track: Dict[str, Any],
    recent_artists: List[str],
    artist_cooldown: int,
) -> float:
    cost = (
        abs(prev_track["tempo"] - next_track["tempo"]) / 35.0
        + 1.3 * abs(prev_track["energy"] - next_track["energy"])
        + 0.9 * abs(prev_track["valence"] - next_track["valence"])
        + 0.6 * abs(prev_track["danceability"] - next_track["danceability"])
        + 0.3 * abs(prev_track["acousticness"] - next_track["acousticness"])
        + 0.2 * abs(prev_track["instrumentalness"] - next_track["instrumentalness"])
        + 0.2 * abs(prev_track["liveness"] - next_track["liveness"])
        + 0.2 * abs(prev_track["speechiness"] - next_track["speechiness"])
    )

    if artist_cooldown > 0 and next_track["artists"] in recent_artists[-artist_cooldown:]:
        cost += 3.0

    return cost


# Orders selected tracks greedily to produce smoother transitions.
def _order_tracks_greedily(
    rows: List[TrackRow],
    playlist_size: int,
    target: Dict[str, float],
    artist_cooldown: int = 0,
) -> List[TrackRow]:
    if not rows:
        return []

    tracks = [_row_to_track(r) for r in rows]
    start = min(tracks, key=lambda t: _distance_to_target(t, target))
    ordered = [start]
    used_ids = {start["id"]}

    while len(ordered) < min(playlist_size, len(tracks)):
        prev = ordered[-1]
        recent_artists = [t["artists"] for t in ordered]
        candidates = [t for t in tracks if t["id"] not in used_ids]
        if not candidates:
            break

        if artist_cooldown > 0:
            allowed = [
                t for t in candidates
                if t["artists"] not in recent_artists[-artist_cooldown:]
            ]
            pool = allowed if allowed else candidates
        else:
            pool = candidates

        nxt = min(
            pool,
            key=lambda t: _transition_cost(
                prev_track=prev,
                next_track=t,
                recent_artists=recent_artists,
                artist_cooldown=artist_cooldown,
            ),
        )
        ordered.append(nxt)
        used_ids.add(nxt["id"])

    return [t["row"] for t in ordered]


# Checks whether a requested genre or style term exists in the database.
def _term_has_genre_or_text_matches(term: str) -> bool:
    cleaned = term.strip()
    if not cleaned:
        return True

    genre_rows = get_candidates_for_feature_ranges(
        valence_range=(0.0, 1.0),
        energy_range=(0.0, 1.0),
        tempo_range=(0.0, 300.0),
        include_genres=[cleaned],
        exclude_genres=None,
        min_popularity=None,
        max_popularity=None,
        year=None,
        year_range=None,
        decade=None,
        exclude_explicit=False,
        exclude_live=False,
        exclude_remix=False,
        exclude_christmas=False,
        duration_range_ms=None,
        limit=1,
    )

    if genre_rows:
        return True
    
    

    text_rows = get_candidates_for_feature_ranges(
        valence_range=(0.0, 1.0),
        energy_range=(0.0, 1.0),
        tempo_range=(0.0, 300.0),
        include_genres=None,
        exclude_genres=None,
        include_text_terms=[cleaned],
        min_popularity=None,
        max_popularity=None,
        year=None,
        year_range=None,
        decade=None,
        exclude_explicit=False,
        exclude_live=False,
        exclude_remix=False,
        exclude_christmas=False,
        duration_range_ms=None,
        limit=1,
    )

    return bool(text_rows)


# Removes requested genre/style terms that cannot be matched in the database.
def _remove_unavailable_genre_terms(
    request: PlaylistRequest,
) -> tuple[PlaylistRequest, List[str]]:
    if not request.include_genres:
        return request, []

    kept_terms: List[str] = []
    missing_terms: List[str] = []

    for term in request.include_genres:
        if _term_has_genre_or_text_matches(term):
            kept_terms.append(term)
        else:
            missing_terms.append(term)

    if not missing_terms:
        return request, []

    updated_request = replace(request)
    updated_request.include_genres = kept_terms

    return updated_request, missing_terms


# Creates a user-facing note for genre/style terms that were ignored.
def _format_unavailable_terms_note(terms: List[str]) -> Optional[str]:
    if not terms:
        return None

    quoted = ", ".join(f"“{term}”" for term in terms)
    label = "term" if len(terms) == 1 else "terms"
    pronoun = "it" if len(terms) == 1 else "them"

    return (
        f"I could not find the requested genre/style {label} {quoted} "
        f"in the database, so I ignored {pronoun} and used the remaining constraints."
    )

# Combines multiple generation notes into one message.
def _combine_generation_notes(*notes: Optional[str]) -> Optional[str]:
    cleaned = [note.strip() for note in notes if note and note.strip()]
    return " ".join(cleaned) if cleaned else None

# Builds a playlist and applies relaxation steps if the strict request is too restrictive.
def build_playlist_with_metadata(
    request: PlaylistRequest,
    debug: bool = False,
) -> PlaylistBuildResult:
    request, unavailable_genre_terms = _remove_unavailable_genre_terms(request)
    unavailable_note = _format_unavailable_terms_note(unavailable_genre_terms)

    track_count = request.resolved_track_count(default=10)

    strict_rows = _build_playlist_once(request, debug=debug)
    if len(strict_rows) >= track_count:
        return PlaylistBuildResult(
            rows=strict_rows[:track_count],
            generation_note=unavailable_note,
        )
    best_rows = strict_rows
    best_note: Optional[str] = None

    relaxation_steps: List[tuple[str, PlaylistRequest]] = []

    if request.min_popularity is not None:
        relaxation_steps.append((
            "lowered the minimum popularity slightly",
            _with_lower_min_popularity(request, 10),
        ))
        relaxation_steps.append((
            "lowered the minimum popularity further",
            _with_lower_min_popularity(request, 20),
        ))

    relaxation_steps.append((
        "relaxed the mood and audio-feature constraints slightly",
        _with_relaxed_mood_constraints(request),
    ))

    if request.min_popularity is not None:
        relaxation_steps.append((
            "relaxed both popularity and mood constraints",
            _with_relaxed_mood_constraints(_with_lower_min_popularity(request, 30)),
        ))

    for note, relaxed_request in relaxation_steps:
        rows = _build_playlist_once(relaxed_request, debug=debug)

        if len(rows) > len(best_rows):
            best_rows = rows
            best_note = note

        if len(rows) >= track_count:
            return PlaylistBuildResult(
                rows=rows[:track_count],
                generation_note=_combine_generation_notes(
                    unavailable_note,
                    f"I relaxed the request slightly to reach {track_count} tracks: I {note}.",
                ),
            )

    if best_rows:
        return PlaylistBuildResult(
            rows=best_rows[:track_count],
            generation_note=_combine_generation_notes(
                unavailable_note,
                (
                    f"I could only find {len(best_rows)} track"
                    f"{'s' if len(best_rows) != 1 else ''} satisfying the hard constraints"
                    + (f"; I only relaxed mood/audio constraints. I {best_note}." if best_note else ".")
                ),
            ),
        )

    return PlaylistBuildResult(
        rows=[],
        generation_note=_combine_generation_notes(
            unavailable_note,
            "I could not find enough matching tracks, even after relaxing the request.",
        ),
    )


# Convenience wrapper that returns only the generated track rows.
def build_playlist_from_request(
    request: PlaylistRequest,
    debug: bool = False,
) -> List[TrackRow]:
    return build_playlist_with_metadata(request, debug=debug).rows


# Runs one playlist-building attempt using the current request constraints.
def _build_playlist_once(
    request: PlaylistRequest,
    debug: bool = False,
) -> List[TrackRow]:
    track_count = request.resolved_track_count(default=10)

    if debug:
        print("\n========== PLAYLIST REQUEST PIPELINE ==========", flush=True)
        print("[INPUT] Original text:", repr(request.original_text), flush=True)
        print("[INPUT] Parsed request:", request.to_dict(), flush=True)
        print("[INPUT] Track count:", track_count, flush=True)

    if track_count >= 25:
        if debug:
            print("[ROUTE] Using large playlist fast path because track_count >= 25", flush=True)
        return _build_large_playlist_fast(request, track_count, debug)

    ranges = _resolve_feature_ranges(request, debug=debug)
    target = _target_profile(ranges)

    required_track_rows = _get_required_track_rows(request, target)
    include_rows = _get_include_rows(request, ranges, target)
    title_seed_rows = _get_title_term_seed_rows(request, ranges, target, track_count)
    style_seed_rows = _get_style_term_seed_rows(
        request,
        ranges,
        target,
        track_count,
    )

    hard_seed_rows = _merge_unique_rows(
        required_track_rows,
        include_rows,
        title_seed_rows,
    )

    max_style_seeds = max(0, track_count - len(hard_seed_rows))
    style_seed_rows = style_seed_rows[:max_style_seeds]

    required_seed_rows = _merge_unique_rows(
        hard_seed_rows,
        style_seed_rows,
    )

    candidate_rows = []

    split_coverage_mode = bool(
        request.include_title_terms
        and (
            request.include_genres
            or request.include_artists
            or request.include_albums
            or request.include_tracks
        )
    )

    used_balanced_genre_terms = len(request.include_genres) > 1 and not split_coverage_mode
    if used_balanced_genre_terms:
        if debug:
            print("[DB] Multiple genre/style terms found. Building balanced candidate pool.")

        candidate_rows = _get_balanced_genre_term_candidates(
            request=request,
            ranges=ranges,
            track_count=track_count,
        )

        if not candidate_rows:
            if debug:
                print("[DB] Balanced genre/style pool failed. Falling back to normal filtered search.")

    if not candidate_rows:
        candidate_rows = get_candidates_for_feature_ranges(
            valence_range=ranges["valence"],
            energy_range=ranges["energy"],
            tempo_range=ranges["tempo"],
            danceability_range=ranges.get("danceability"),
            acousticness_range=ranges.get("acousticness"),
            instrumentalness_range=ranges.get("instrumentalness"),
            liveness_range=ranges.get("liveness"),
            speechiness_range=ranges.get("speechiness"),
            exclude_artists=request.exclude_artists,
            include_albums=None,
            exclude_albums=request.exclude_albums,
            include_tracks=None,
            exclude_tracks=request.exclude_tracks,
            include_title_terms=None if split_coverage_mode else request.include_title_terms,
            exclude_title_terms=request.exclude_title_terms,

            include_genres=None if split_coverage_mode else request.include_genres,
            exclude_genres=request.exclude_genres,
            min_popularity=request.min_popularity,
            max_popularity=request.max_popularity,
            year=request.year,
            year_range=request.year_range,
            decade=request.decade,
            exclude_explicit=request.exclude_explicit,
            exclude_live=request.exclude_live,
            exclude_remix=request.exclude_remix,
            exclude_christmas=request.exclude_christmas,
            duration_range_ms=request.duration_range_ms,
            limit=max(100, track_count * 10),
        )

    
    candidate_rows = _merge_unique_rows(required_seed_rows, candidate_rows)
    if not candidate_rows:
        if debug:
            print("[DB] No candidates found, trying broader filtered fallback.")
        candidate_rows = _get_broad_fallback_candidates(request, track_count)

    if not candidate_rows and request.include_genres and not used_balanced_genre_terms:
        if debug:
            print("[DB] No genre-column matches. Trying requested genre/style terms as metadata text search.")
        candidate_rows = _get_text_term_fallback_candidates(request, track_count)

    if not candidate_rows:
        if debug:
            print("[DB] No filtered fallback candidates found. Returning no matches instead of random unrelated tracks.")
        return []

    facts_path = write_rows_to_facts(
        rows=candidate_rows,
        request=request,
        valence_range=ranges["valence"],
        energy_range=ranges["energy"],
        tempo_range=ranges["tempo"],
        include_artists=request.include_artists,
        include_albums=request.include_albums,
        include_tracks=request.include_tracks,
        debug=debug,
    )

    rules_path = _resolve_rules_path()

    if debug:
        print("[ASP] Starting clingo solve...")
    selected_ids = run_clingo(
        facts_file=facts_path,
        rules_file=rules_path,
        playlist_size=track_count,
    )
    if debug:
        print("[ASP] Clingo solve finished.")
        print(f"[ASP] Selected {len(selected_ids)} track IDs.")

    selected_rows = _fetch_rows_by_ids(selected_ids)

    if not selected_rows:
        if debug:
            print("[ASP] Solver returned no playlist, using pinned-track fallback.")
        selected_rows = _select_with_required_seed_rows(
            candidate_rows,
            required_seed_rows,
            track_count,
            target,
        )
    elif required_seed_rows:
        selected_rows = _select_with_required_seed_rows(
            selected_rows,
            required_seed_rows,
            track_count,
            target,
        )
    
    ordered_rows = _order_tracks_greedily(
        rows=selected_rows,
        playlist_size=track_count,
        target=target,
        artist_cooldown=request.resolved_artist_gap(track_count),
    )

    if debug:
        _debug_print_playlist(ordered_rows, "[ORDER] Final ordered playlist")

    return ordered_rows


# Uses a faster candidate-selection path for larger playlists.
def _build_large_playlist_fast(request, track_count, debug=False):
    ranges = _resolve_feature_ranges(request, debug=debug)
    target = _target_profile(ranges)

    required_track_rows = _get_required_track_rows(request, target)
    include_rows = _get_include_rows(request, ranges, target)

    candidate_rows = get_candidates_for_feature_ranges(
        valence_range=ranges["valence"],
        energy_range=ranges["energy"],
        tempo_range=ranges["tempo"],
        danceability_range=ranges.get("danceability"),
        acousticness_range=ranges.get("acousticness"),
        instrumentalness_range=ranges.get("instrumentalness"),
        liveness_range=ranges.get("liveness"),
        speechiness_range=ranges.get("speechiness"),
        exclude_artists=request.exclude_artists,
        exclude_albums=request.exclude_albums,
        exclude_tracks=request.exclude_tracks,
        include_title_terms=request.include_title_terms,
        exclude_title_terms=request.exclude_title_terms,
        include_genres=request.include_genres,
        exclude_genres=request.exclude_genres,
        min_popularity=request.min_popularity,
        max_popularity=request.max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        limit=max(300, track_count * 8),
    )

    candidate_rows = _merge_unique_rows(
        required_track_rows,
        include_rows,
        candidate_rows,
    )

    selected = _select_with_required_seed_rows(
        candidate_rows,
        _merge_unique_rows(required_track_rows, include_rows),
        track_count,
        target,
    )

    return _order_tracks_greedily(
        selected,
        playlist_size=track_count,
        target=target,
        artist_cooldown=request.resolved_artist_gap(track_count),
    )

# Keeps required seed tracks and fills the remaining playlist slots with best matches.
def _select_with_required_seed_rows(
    rows: List[TrackRow],
    seed_rows: List[TrackRow],
    track_count: int,
    target: Dict[str, float],
) -> List[TrackRow]:
    seed_rows = _merge_unique_rows(seed_rows)

    if len(seed_rows) > track_count:
        seed_rows = seed_rows[:track_count]

    seed_ids = {row[0] for row in seed_rows}
    remaining_rows = [row for row in rows if row[0] not in seed_ids]

    filler_rows = _fallback_select(
        remaining_rows,
        track_count - len(seed_rows),
        target,
    )

    return _merge_unique_rows(seed_rows, filler_rows)


# Creates a relaxed copy of the request with a lower minimum popularity threshold.
def _with_lower_min_popularity(request: PlaylistRequest, amount: int) -> PlaylistRequest:
    relaxed = replace(request)
    if relaxed.min_popularity is not None:
        relaxed.min_popularity = max(0, relaxed.min_popularity - amount)
    return relaxed

# Creates a relaxed copy of the request with mood and audio-feature constraints removed.
def _with_relaxed_mood_constraints(request: PlaylistRequest) -> PlaylistRequest:
    relaxed = replace(request)
    relaxed.activity = None
    relaxed.emotions = []
    relaxed.emotion_weights = {}
    relaxed.tempo_range = None
    relaxed.energy_range = None
    relaxed.valence_range = None
    relaxed.danceability_range = None
    relaxed.acousticness_range = None
    relaxed.instrumentalness_range = None
    relaxed.liveness_range = None
    relaxed.speechiness_range = None
    setattr(relaxed, "_ignore_original_emotion_text", True)
    return relaxed


# Resolves final feature ranges from explicit ranges, emotion weights, and activity presets.
def _resolve_feature_ranges(request: PlaylistRequest, debug: bool = False) -> Dict[str, Optional[Range]]:
    emotion_text: Optional[str] = None

    if getattr(request, "emotion_weights", None):
        emotion_weights = request.emotion_weights
        if debug:
            print("[EMOTION] Using LLM-supplied emotion weights:", emotion_weights)

    elif getattr(request, "_ignore_original_emotion_text", False):
        emotion_weights = {}
        if debug:
            print("[EMOTION] Mood relaxation enabled; ignoring original emotion text.")

    else:
        emotion_text = " ".join(request.emotions) if request.emotions else request.original_text
        emotion_weights = text_to_emotion_weights(emotion_text, debug=debug)

    emotion_ranges = compute_feature_ranges(emotion_weights, debug=debug) if emotion_weights else {}
    activity_ranges = _ACTIVITY_PRESETS.get((request.activity or "").lower(), {})

    ranges: Dict[str, Optional[Range]] = {
        "valence": request.valence_range or emotion_ranges.get("valence") or (0.0, 1.0),
        "energy": request.energy_range or activity_ranges.get("energy") or emotion_ranges.get("energy") or (0.0, 1.0),
        "tempo": request.tempo_range or activity_ranges.get("tempo") or emotion_ranges.get("tempo") or (60.0, 180.0),
        "danceability": request.danceability_range or activity_ranges.get("danceability") or emotion_ranges.get("danceability"),
        "acousticness": request.acousticness_range or activity_ranges.get("acousticness") or emotion_ranges.get("acousticness"),
        "instrumentalness": request.instrumentalness_range or activity_ranges.get("instrumentalness") or emotion_ranges.get("instrumentalness"),
        "liveness": request.liveness_range or activity_ranges.get("liveness") or emotion_ranges.get("liveness"),
        "speechiness": request.speechiness_range or activity_ranges.get("speechiness") or emotion_ranges.get("speechiness"),
    }

    if debug:
        if emotion_text is not None:
            print("[EMOTION] Emotion text used:", repr(emotion_text))

        print("\n[RANGES] Final ranges passed into DB/ASP:")
        for key, value in ranges.items():
            print(f"  {key:16}: {value}")

    return ranges

# Retrieves seed rows for explicitly included artists, albums, and tracks.
def _get_include_rows(
    request: PlaylistRequest,
    ranges: Dict[str, Optional[Range]],
    target: Dict[str, float],
) -> List[TrackRow]:
    groups: List[List[TrackRow]] = []

    common_kwargs = dict(
        valence_range=ranges["valence"],
        energy_range=ranges["energy"],
        tempo_range=ranges["tempo"],
        danceability_range=ranges.get("danceability"),
        acousticness_range=ranges.get("acousticness"),
        instrumentalness_range=ranges.get("instrumentalness"),
        liveness_range=ranges.get("liveness"),
        speechiness_range=ranges.get("speechiness"),
        include_title_terms=request.include_title_terms,
        exclude_title_terms=request.exclude_title_terms,
        include_genres=request.include_genres,
        exclude_genres=request.exclude_genres,
        min_popularity=request.min_popularity,
        max_popularity=request.max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
    )

    common_broad_kwargs = dict(
        include_genres=request.include_genres,
        exclude_genres=request.exclude_genres,
        min_popularity=request.min_popularity,
        max_popularity=request.max_popularity,
        include_title_terms=request.include_title_terms,
        exclude_title_terms=request.exclude_title_terms,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        ignore_feature_ranges=True,
    )

    seed_broad_kwargs = dict(
        common_broad_kwargs,

        exclude_genres=request.exclude_genres,

        include_title_terms=None,
        exclude_title_terms=None,

        min_popularity=None,
        max_popularity=None,

        ignore_feature_ranges=True,
    )

    if request.include_artists:
        ranged_rows = get_tracks_for_artists(
            request.include_artists,
            limit_per_artist=20,
            ignore_feature_ranges=False,
            **common_kwargs,
        )

        broad_rows = get_tracks_for_artists(
            request.include_artists,
            limit_per_artist=20,
            **seed_broad_kwargs,
        )

        artist_rows = _merge_unique_rows(ranged_rows, broad_rows)
        track_count = request.resolved_track_count(default=10)
        artist_target_count = request.resolved_artist_target_count(track_count)

        chosen_artist_rows = _choose_random_pool_per_requested_artist(
            requested_artists=request.include_artists,
            rows=artist_rows,
            target=target,
            per_artist=artist_target_count,
        )

        groups.append(chosen_artist_rows)

    if request.include_albums:
        ranged_rows = get_tracks_for_albums(
            request.include_albums,
            limit_per_album=30,
            ignore_feature_ranges=False,
            **common_kwargs,
        )
        broad_rows = get_tracks_for_albums(
            request.include_albums,
            limit_per_album=20,
            **seed_broad_kwargs,
        )
        groups.append(_merge_unique_rows(ranged_rows, broad_rows))

    if request.include_tracks:
        ranged_rows = get_tracks_by_names(
            request.include_tracks,
            limit_per_name=12,
            ignore_feature_ranges=False,
            **common_kwargs,
        )
        broad_rows = get_tracks_by_names(
            request.include_tracks,
            limit_per_name=12,
            **seed_broad_kwargs,
        )
        groups.append(_choose_best_per_requested_name(request.include_tracks, _merge_unique_rows(ranged_rows, broad_rows), target, field_index=1))

    return _merge_unique_rows(*groups)


# Chooses the best matching database row for each requested name.
def _choose_best_per_requested_name(
    requested_names: List[str],
    rows: List[TrackRow],
    target: Dict[str, float],
    *,
    field_index: int,
) -> List[TrackRow]:
    chosen: List[TrackRow] = []
    seen_ids = set()

    for requested in requested_names:
        if field_index == 2:
            matching = [row for row in rows if _artist_exact_match(requested, row[2])]
        else:
            matching = [row for row in rows if requested.lower() in str(row[field_index]).lower()]

        if not matching:
            continue

        ranked = sorted(
            matching,
            key=lambda row: _rank_rows_for_request(rows, target)
        )

        pool = ranked[: min(5, len(ranked))]
        best = random.choice(pool)

        if best[0] in seen_ids:
            continue

        chosen.append(best)
        seen_ids.add(best[0])

    return chosen

# Randomly selects a small pool of suitable tracks for each requested artist.
def _choose_random_pool_per_requested_artist(
    requested_artists: List[str],
    rows: List[TrackRow],
    target: Dict[str, float],
    *,
    per_artist: int = 4,
) -> List[TrackRow]:
    chosen: List[TrackRow] = []
    seen_ids = set()

    for artist in requested_artists:
        matching = [
            row for row in rows
            if _artist_exact_match(artist, row[2])
        ]

        if not matching:
            continue

        ranked = sorted(
            matching,
            key=lambda row: _rank_rows_for_request(rows, target)
        )

        pool = ranked[: min(12, len(ranked))]
        sample = random.sample(pool, k=min(per_artist, len(pool)))

        for row in sample:
            if row[0] not in seen_ids:
                chosen.append(row)
                seen_ids.add(row[0])

    return chosen

# Merges track row lists while removing duplicate track IDs.
def _merge_unique_rows(*groups: List[TrackRow]) -> List[TrackRow]:
    rows: List[TrackRow] = []
    seen_ids = set()
    for group in groups:
        for row in group:
            if row[0] in seen_ids:
                continue
            rows.append(row)
            seen_ids.add(row[0])
    return rows


# Retrieves broader fallback candidates when strict filtering returns too few tracks.
def _get_broad_fallback_candidates(request: PlaylistRequest, track_count: int) -> List[TrackRow]:
    return get_candidates_for_feature_ranges(
        valence_range=(0.0, 1.0),
        energy_range=(0.0, 1.0),
        tempo_range=(60.0, 180.0),
        exclude_artists=request.exclude_artists,
        include_albums=request.include_albums,
        exclude_albums=request.exclude_albums,
        include_tracks=request.include_tracks,
        exclude_tracks=request.exclude_tracks,
        include_title_terms=request.include_title_terms,
        exclude_title_terms=request.exclude_title_terms,  
        include_genres=request.include_genres,
        exclude_genres=request.exclude_genres,
        min_popularity=request.min_popularity,
        max_popularity=request.max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        limit=max(track_count * 3, 50),
    )

TEXT_FALLBACK_BLOCKLIST = {
    "pop",
    "rap",
    "rock",
    "jazz",
    "folk",
    "r&b",
    "rnb",
    "hip-hop",
    "hip hop",
    "dance",
    "metal",
    "soul",
}

# Searches requested style terms as metadata text when they are not real database genres.
def _get_text_term_fallback_candidates(
    request: PlaylistRequest,
    track_count: int,
) -> List[TrackRow]:
    text_terms = [
        term.strip()
        for term in request.include_genres
        if term.strip()
        and term.strip().lower() not in TEXT_FALLBACK_BLOCKLIST
    ]
    if not text_terms:
        return []

    return get_candidates_for_feature_ranges(
        valence_range=(0.0, 1.0),
        energy_range=(0.0, 1.0),
        tempo_range=(60.0, 180.0),
        exclude_artists=request.exclude_artists,
        include_albums=request.include_albums,
        exclude_albums=request.exclude_albums,
        include_tracks=request.include_tracks,
        exclude_tracks=request.exclude_tracks,

        include_title_terms=request.include_title_terms,
        exclude_title_terms=request.exclude_title_terms,
        include_genres=None,
        exclude_genres=request.exclude_genres,

        include_text_terms=text_terms,

        min_popularity=request.min_popularity,
        max_popularity=request.max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        limit=max(track_count * 5, 80),
    )


# Builds a balanced candidate pool when the request includes multiple genre/style terms.
def _get_balanced_genre_term_candidates(
    request: PlaylistRequest,
    ranges: Dict[str, Optional[Range]],
    track_count: int,
) -> List[TrackRow]:
    terms = [term.strip() for term in request.include_genres if term.strip()]
    if not terms:
        return []

    per_term_limit = max(20, track_count * 4)
    groups: List[List[TrackRow]] = []

    found_text_term = False
    had_text_term_candidate = False

    for term in terms:
        genre_rows = get_candidates_for_feature_ranges(
            valence_range=ranges["valence"],
            energy_range=ranges["energy"],
            tempo_range=ranges["tempo"],
            danceability_range=ranges.get("danceability"),
            acousticness_range=ranges.get("acousticness"),
            instrumentalness_range=ranges.get("instrumentalness"),
            liveness_range=ranges.get("liveness"),
            speechiness_range=ranges.get("speechiness"),
            exclude_artists=request.exclude_artists,
            include_albums=None,
            exclude_albums=request.exclude_albums,
            include_tracks=None,
            exclude_tracks=request.exclude_tracks,
            include_title_terms=None,
            exclude_title_terms=request.exclude_title_terms,
            include_genres=[term],
            exclude_genres=request.exclude_genres,
            min_popularity=request.min_popularity,
            max_popularity=request.max_popularity,
            year=request.year,
            year_range=request.year_range,
            decade=request.decade,
            exclude_explicit=request.exclude_explicit,
            exclude_live=request.exclude_live,
            exclude_remix=request.exclude_remix,
            exclude_christmas=request.exclude_christmas,
            duration_range_ms=request.duration_range_ms,
            limit=per_term_limit,
        )

        if genre_rows:
            found_real_genre = True
            groups.append(genre_rows)
            continue

        if term.strip().lower() in TEXT_FALLBACK_BLOCKLIST:
            continue

    

        text_rows = get_candidates_for_feature_ranges(
            valence_range=ranges["valence"],
            energy_range=ranges["energy"],
            tempo_range=ranges["tempo"],
            danceability_range=ranges.get("danceability"),
            acousticness_range=ranges.get("acousticness"),
            instrumentalness_range=ranges.get("instrumentalness"),
            liveness_range=ranges.get("liveness"),
            speechiness_range=ranges.get("speechiness"),
            exclude_artists=request.exclude_artists,
            include_albums=None,
            exclude_albums=request.exclude_albums,
            include_tracks=None,
            exclude_tracks=request.exclude_tracks,
            include_title_terms=None,
            exclude_title_terms=request.exclude_title_terms,
            include_genres=None,
            exclude_genres=request.exclude_genres,
            include_text_terms=[term],
            min_popularity=request.min_popularity,
            max_popularity=request.max_popularity,
            year=request.year,
            year_range=request.year_range,
            decade=request.decade,
            exclude_explicit=request.exclude_explicit,
            exclude_live=request.exclude_live,
            exclude_remix=request.exclude_remix,
            exclude_christmas=request.exclude_christmas,
            duration_range_ms=request.duration_range_ms,
            limit=per_term_limit,
        )

        if not text_rows:
            text_rows = get_candidates_for_feature_ranges(
                valence_range=(0.0, 1.0),
                energy_range=(0.0, 1.0),
                tempo_range=(0.0, 300.0),
                danceability_range=None,
                acousticness_range=None,
                instrumentalness_range=None,
                liveness_range=None,
                speechiness_range=None,
                exclude_artists=request.exclude_artists,
                include_albums=None,
                exclude_albums=request.exclude_albums,
                include_tracks=None,
                exclude_tracks=request.exclude_tracks,
                include_title_terms=None,
                exclude_title_terms=request.exclude_title_terms,
                include_genres=None,
                exclude_genres=request.exclude_genres,
                include_text_terms=[term],
                min_popularity=None,
                max_popularity=None,
                year=request.year,
                year_range=request.year_range,
                decade=request.decade,
                exclude_explicit=request.exclude_explicit,
                exclude_live=request.exclude_live,
                exclude_remix=request.exclude_remix,
                exclude_christmas=request.exclude_christmas,
                duration_range_ms=request.duration_range_ms,
                limit=per_term_limit,
            )

        if text_rows:
            found_text_term = True
            groups.append(text_rows)

    if not groups:
        return []


    if had_text_term_candidate and not found_text_term:
        return []

    balanced: List[TrackRow] = []
    seen_ids = set()

    max_len = max(len(group) for group in groups)
    for i in range(max_len):
        for group in groups:
            if i >= len(group):
                continue

            row = group[i]
            if row[0] in seen_ids:
                continue

            balanced.append(row)
            seen_ids.add(row[0])

            if len(balanced) >= max(track_count * 10, 100):
                return balanced

    return balanced

#Decides how many seed tracks should represent each title or style term.
def _coverage_seed_count(track_count: int) -> int:
    return max(2, min(4, track_count // 3))

# Ranks rows by closeness to the target profile, with a small popularity bonus.
def _rank_rows_for_request(
    rows: List[TrackRow],
    target: Dict[str, float],
) -> List[TrackRow]:
    return sorted(
        rows,
        key=lambda row: (
            _distance_to_target(_row_to_track(row), target)
            - 0.45 * ((row[15] or 0) / 100.0)
        ),
    )

# Retrieves seed tracks for requested title terms.
def _get_title_term_seed_rows(
    request: PlaylistRequest,
    ranges: Dict[str, Optional[Range]],
    target: Dict[str, float],
    track_count: int,
) -> List[TrackRow]:
    seed_rows: List[TrackRow] = []
    seen_ids = set()
    per_term = _coverage_seed_count(track_count)

    for term in request.include_title_terms:
        cleaned = term.strip()
        if not cleaned:
            continue

        rows = _get_rows_for_one_title_term(
            request=request,
            term=cleaned,
            ranges=ranges,
            broad=False,
        )

        if not rows:
            rows = _get_rows_for_one_title_term(
                request=request,
                term=cleaned,
                ranges=ranges,
                broad=True,
            )

        added_for_term = 0

        for row in _rank_rows_for_request(rows, target):
            if row[0] in seen_ids:
                continue

            seed_rows.append(row)
            seen_ids.add(row[0])
            added_for_term += 1

            if added_for_term >= per_term:
                break

    return seed_rows


# Retrieves rows for one requested title term using strict or broad matching.
def _get_rows_for_one_title_term(
    request: PlaylistRequest,
    term: str,
    ranges: Dict[str, Optional[Range]],
    *,
    broad: bool,
) -> List[TrackRow]:
    if broad:
        valence_range = (0.0, 1.0)
        energy_range = (0.0, 1.0)
        tempo_range = (0.0, 300.0)
        danceability_range = None
        acousticness_range = None
        instrumentalness_range = None
        liveness_range = None
        speechiness_range = None
        min_popularity = None
        max_popularity = None
    else:
        valence_range = ranges["valence"]
        energy_range = ranges["energy"]
        tempo_range = ranges["tempo"]
        danceability_range = ranges.get("danceability")
        acousticness_range = ranges.get("acousticness")
        instrumentalness_range = ranges.get("instrumentalness")
        liveness_range = ranges.get("liveness")
        speechiness_range = ranges.get("speechiness")
        min_popularity = request.min_popularity
        max_popularity = request.max_popularity

    return get_candidates_for_feature_ranges(
        valence_range=valence_range,
        energy_range=energy_range,
        tempo_range=tempo_range,
        danceability_range=danceability_range,
        acousticness_range=acousticness_range,
        instrumentalness_range=instrumentalness_range,
        liveness_range=liveness_range,
        speechiness_range=speechiness_range,
        exclude_artists=request.exclude_artists,
        include_albums=None,
        exclude_albums=request.exclude_albums,
        include_tracks=None,
        exclude_tracks=request.exclude_tracks,

        include_title_terms=[term],
        exclude_title_terms=request.exclude_title_terms,


        include_genres=None,
        exclude_genres=request.exclude_genres,

        min_popularity=min_popularity,
        max_popularity=max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        limit=30,
    )


# Retrieves seed tracks for requested genre or style terms.
def _get_style_term_seed_rows(
    request: PlaylistRequest,
    ranges: Dict[str, Optional[Range]],
    target: Dict[str, float],
    track_count: int,
) -> List[TrackRow]:
    seed_rows: List[TrackRow] = []
    seen_ids = set()
    per_term = _coverage_seed_count(track_count)

    for term in request.include_genres:
        cleaned = term.strip()
        if not cleaned:
            continue

        rows = _get_rows_for_one_style_term(
            request=request,
            term=cleaned,
            ranges=ranges,
            broad=False,
        )

        if not rows:
            rows = _get_rows_for_one_style_term(
                request=request,
                term=cleaned,
                ranges=ranges,
                broad=True,
            )

        added_for_term = 0

        for row in _rank_rows_for_request(rows, target):
            if row[0] in seen_ids:
                continue

            seed_rows.append(row)
            seen_ids.add(row[0])
            added_for_term += 1

            if added_for_term >= per_term:
                break

    return seed_rows


# Retrieves rows for one requested genre/style term using genre or metadata matching.
def _get_rows_for_one_style_term(
    request: PlaylistRequest,
    term: str,
    ranges: Dict[str, Optional[Range]],
    *,
    broad: bool,
) -> List[TrackRow]:
    if broad:
        valence_range = (0.0, 1.0)
        energy_range = (0.0, 1.0)
        tempo_range = (0.0, 300.0)
        danceability_range = None
        acousticness_range = None
        instrumentalness_range = None
        liveness_range = None
        speechiness_range = None
        min_popularity = None
        max_popularity = None
    else:
        valence_range = ranges["valence"]
        energy_range = ranges["energy"]
        tempo_range = ranges["tempo"]
        danceability_range = ranges.get("danceability")
        acousticness_range = ranges.get("acousticness")
        instrumentalness_range = ranges.get("instrumentalness")
        liveness_range = ranges.get("liveness")
        speechiness_range = ranges.get("speechiness")
        min_popularity = request.min_popularity
        max_popularity = request.max_popularity

 
    genre_rows = get_candidates_for_feature_ranges(
        valence_range=valence_range,
        energy_range=energy_range,
        tempo_range=tempo_range,
        danceability_range=danceability_range,
        acousticness_range=acousticness_range,
        instrumentalness_range=instrumentalness_range,
        liveness_range=liveness_range,
        speechiness_range=speechiness_range,
        exclude_artists=request.exclude_artists,
        include_albums=None,
        exclude_albums=request.exclude_albums,
        include_tracks=None,
        exclude_tracks=request.exclude_tracks,
        include_title_terms=None,
        exclude_title_terms=request.exclude_title_terms,
        include_genres=[term],
        exclude_genres=request.exclude_genres,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        limit=30,
    )

    if genre_rows:
        return genre_rows

    if term.strip().lower() in TEXT_FALLBACK_BLOCKLIST:
        return []


    return get_candidates_for_feature_ranges(
        valence_range=valence_range,
        energy_range=energy_range,
        tempo_range=tempo_range,
        danceability_range=danceability_range,
        acousticness_range=acousticness_range,
        instrumentalness_range=instrumentalness_range,
        liveness_range=liveness_range,
        speechiness_range=speechiness_range,
        exclude_artists=request.exclude_artists,
        include_albums=None,
        exclude_albums=request.exclude_albums,
        include_tracks=None,
        exclude_tracks=request.exclude_tracks,
        include_genres=None,
        exclude_genres=request.exclude_genres,
        include_text_terms=[term],
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        year=request.year,
        year_range=request.year_range,
        decade=request.decade,
        exclude_explicit=request.exclude_explicit,
        exclude_live=request.exclude_live,
        exclude_remix=request.exclude_remix,
        exclude_christmas=request.exclude_christmas,
        duration_range_ms=request.duration_range_ms,
        limit=30,
    )


# Finds the ASP rules file used by clingo.
def _resolve_rules_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "emotion_playlist.lp",
        here / "rules" / "emotion_playlist.lp",
        here.parent / "emotion_playlist.lp",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find emotion_playlist.lp")


# Fetches full database rows for the track IDs selected by the solver.
def _fetch_rows_by_ids(track_ids: List[str]) -> List[TrackRow]:
    return get_tracks_by_ids(track_ids)


# Selects the best fallback tracks when the ASP solver does not return a result.
def _fallback_select(rows: List[TrackRow], track_count: int, target: Dict[str, float]) -> List[TrackRow]:
    ordered = sorted(rows, key=lambda row: _rank_rows_for_request(rows, target))
    return ordered[:track_count]

# Retrieves explicitly required tracks while ignoring mood and popularity filters.
def _get_required_track_rows(
    request: PlaylistRequest,
    target: Dict[str, float],
) -> List[TrackRow]:
    if not request.include_tracks:
        return []

    rows = get_tracks_by_names(
        request.include_tracks,
        limit_per_name=20,
        valence_range=None,
        energy_range=None,
        tempo_range=None,
        danceability_range=None,
        acousticness_range=None,
        instrumentalness_range=None,
        liveness_range=None,
        speechiness_range=None,
        include_genres=None,
        exclude_genres=None,
        min_popularity=None,
        max_popularity=None,
        year=None,
        year_range=None,
        decade=None,
        exclude_explicit=False,
        exclude_live=False,
        exclude_remix=False,
        exclude_christmas=False,
        duration_range_ms=None,
        ignore_feature_ranges=True,
    )

    missing = [
        name
        for name in request.include_tracks
        if not any(name.lower() in str(row[1] or "").lower() for row in rows)
    ]

    if missing:
        raise ValueError(
            "Could not find required track(s) in the dataset: "
            + ", ".join(missing)
        )

    return _choose_best_per_requested_name(
        request.include_tracks,
        rows,
        target,
        field_index=1,
    )

# Prints the final playlist and feature values for debugging.
def _debug_print_playlist(rows: List[TrackRow], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for row in rows:
        print(
            f"- {row[1]} — {row[2]}\n"
            f"    album={row[11]}, year={row[14]}, popularity={row[15]}, genre={row[16]}, duration_ms={row[13]}, "
            f"tempo={row[3]:.1f}, energy={row[4]:.2f}, valence={row[5]:.2f}, "
            f"danceability={row[6]:.2f}, acousticness={row[7]:.2f}, "
            f"instrumentalness={row[8]:.2f}, liveness={row[9]:.2f}, speechiness={row[10]:.2f}"
        )


# Splits a stored artist string into individual lowercase artist names.
def _split_artist_names(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if ";" in text:
        return [part.strip().lower() for part in text.split(";") if part.strip()]

    if "," in text:
        return [part.strip().lower() for part in text.split(",") if part.strip()]

    return [text.lower()]

# Checks whether a requested artist exactly matches one of a track's artists.
def _artist_exact_match(requested_artist: str, artists_value: object) -> bool:
    requested = requested_artist.strip().lower()
    if not requested:
        return False
    return requested in _split_artist_names(artists_value)
