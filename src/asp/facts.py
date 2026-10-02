from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.data.db import TrackRow
from src.nlp.request_schema import PlaylistRequest, Range

# Converts database track rows and parsed playlist constraints into ASP facts.
# These facts are written to a .lp file so that clingo can reason over the
# candidate tracks and select a playlist.

FACTS_DIR = Path(__file__).resolve().parent / "asp_facts"
FACTS_DIR.mkdir(parents=True, exist_ok=True)

# Converts candidate database rows into song/12 and performer/2 ASP facts.
def rows_to_facts(rows: Iterable[TrackRow]) -> str:
    lines: List[str] = []

    for row in rows:
        track_id = row[0]
        name = row[1]
        artists = row[2]
        tempo = row[3]
        energy = row[4]
        valence = row[5]
        danceability = row[6]
        acousticness = row[7]
        instrumentalness = row[8]
        liveness = row[9]
        speechiness = row[10]
        album = row[11]

        line = (
            f'song("{_esc(track_id)}", {int(round(tempo or 0))}, '
            f'{int(round((energy or 0) * 100))}, {int(round((valence or 0) * 100))}, '
            f'{int(round((danceability or 0) * 100))}, {int(round((acousticness or 0) * 100))}, '
            f'{int(round((instrumentalness or 0) * 100))}, {int(round((liveness or 0) * 100))}, '
            f'{int(round((speechiness or 0) * 100))}, "{_esc(name)}", "{_esc(str(artists))}", "{_esc(str(album or ''))}").'
        )
        lines.append(line)

        for artist in _split_artists(artists):
            lines.append(f'performer("{_esc(artist)}", "{_esc(track_id)}").')

    return "\n".join(lines)


# Converts the target feature ranges into ASP facts for valence, energy, and tempo.
def targets_to_facts(
    valence_range: Range,
    energy_range: Range,
    tempo_range: Range,
) -> str:
    target_valence = int(round(((valence_range[0] + valence_range[1]) / 2.0) * 100))
    target_energy = int(round(((energy_range[0] + energy_range[1]) / 2.0) * 100))
    target_tempo = int(round((tempo_range[0] + tempo_range[1]) / 2.0))

    return "\n".join(
        [
            f"target_valence({target_valence}).",
            f"target_energy({target_energy}).",
            f"target_tempo({target_tempo}).",
        ]
    )

# Converts request-specific constraints into ASP facts such as required artists,
# banned artists, required tracks, banned tracks, and matching relationships.
def request_to_facts(
    request: PlaylistRequest,
    rows: Sequence[TrackRow],
    track_count: int,
    include_artists: Optional[Sequence[str]] = None,
    include_albums: Optional[Sequence[str]] = None,
    include_tracks: Optional[Sequence[str]] = None,
) -> str:
    lines: List[str] = []

    lines.append(f"artist_gap({request.resolved_artist_gap(track_count)}).")

    if request.max_tracks_per_artist is not None and request.max_tracks_per_artist > 0:
        lines.append(f"max_tracks_per_artist({request.max_tracks_per_artist}).")

    requested_includes_artists = list(include_artists or request.include_artists)
    requested_excludes_artists = list(request.exclude_artists)
    if requested_includes_artists:
        target_count = request.resolved_artist_target_count(track_count)
        if target_count > 0:
            lines.append(f"artist_target_count({target_count}).")
            
    requested_includes_albums = list(include_albums or request.include_albums)
    requested_excludes_albums = list(request.exclude_albums)
    requested_includes_tracks = list(include_tracks or request.include_tracks)
    requested_excludes_tracks = list(request.exclude_tracks)

    for artist in requested_includes_artists:
        cleaned = artist.strip()
        if cleaned:
            lines.append(f'required_artist("{_esc(cleaned)}").')

    for artist in requested_excludes_artists:
        cleaned = artist.strip()
        if cleaned:
            lines.append(f'banned_artist("{_esc(cleaned)}").')

    for album in requested_includes_albums:
        cleaned = album.strip()
        if cleaned:
            lines.append(f'required_album("{_esc(cleaned)}").')

    for album in requested_excludes_albums:
        cleaned = album.strip()
        if cleaned:
            lines.append(f'banned_album("{_esc(cleaned)}").')

    for track in requested_includes_tracks:
        cleaned = track.strip()
        if cleaned:
            lines.append(f'required_track("{_esc(cleaned)}").')

    for track in requested_excludes_tracks:
        cleaned = track.strip()
        if cleaned:
            lines.append(f'banned_track("{_esc(cleaned)}").')

    tracked_artists = sorted({a.strip() for a in requested_includes_artists + requested_excludes_artists if a.strip()})
    tracked_albums = sorted({a.strip() for a in requested_includes_albums + requested_excludes_albums if a.strip()})
    tracked_tracks = sorted({t.strip() for t in requested_includes_tracks + requested_excludes_tracks if t.strip()})

    for row in rows:
        track_id = row[0]
        name = str(row[1] or "")
        artists = str(row[2] or "")
        album = str(row[11] or "")

        for artist in tracked_artists:
            if _artist_exact_match(artist, artists):
                lines.append(f'artist_match("{_esc(artist)}", "{_esc(track_id)}").')
        for album_name in tracked_albums:
            if album_name.lower() in album.lower():
                lines.append(f'album_match("{_esc(album_name)}", "{_esc(track_id)}").')
        for track_name in tracked_tracks:
            if track_name.lower() in name.lower():
                lines.append(f'track_match("{_esc(track_name)}", "{_esc(track_id)}").')

    return "\n".join(lines)


# Writes all generated ASP facts for the current playlist request to a .lp file.
def write_rows_to_facts(
    rows: Sequence[TrackRow],
    request: PlaylistRequest,
    valence_range: Range,
    energy_range: Range,
    tempo_range: Range,
    *,
    include_artists: Optional[Sequence[str]] = None,
    include_albums: Optional[Sequence[str]] = None,
    include_tracks: Optional[Sequence[str]] = None,
    filename: str = "playlist_tracks.lp",
    debug: bool = False,
) -> Path:
    rows = list(rows)

    sections = [
        rows_to_facts(rows),
        targets_to_facts(
            valence_range=valence_range,
            energy_range=energy_range,
            tempo_range=tempo_range,
        ),
        request_to_facts(
            request=request,
            rows=rows,
            track_count=request.resolved_track_count(),
            include_artists=include_artists,
            include_albums=include_albums,
            include_tracks=include_tracks,
        ),
    ]

    facts_str = "\n\n".join(section for section in sections if section.strip())
    out_path = FACTS_DIR / filename
    out_path.write_text(facts_str, encoding="utf-8")

    if debug:
        print(f"[ASP] Wrote {len(rows)} candidates + target/request facts to {out_path}")

    return out_path


# Splits a stored artist string into individual artist names.
def _split_artists(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]

    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]

    return [text]


# Escapes text so it can be safely written inside ASP string facts.
def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


# Checks whether a requested artist exactly matches one of the artists on a track.
def _artist_exact_match(requested_artist: str, artists_value: object) -> bool:
    requested = requested_artist.strip().lower()
    if not requested:
        return False
    return requested in [a.strip().lower() for a in _split_artists(artists_value)]
