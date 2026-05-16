from __future__ import annotations

import ast
import re
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TrackRow = Tuple[
    str,   # id
    str,   # name
    str,   # artists
    float, # tempo
    float, # energy
    float, # valence
    float, # danceability
    float, # acousticness
    float, # instrumentalness
    float, # liveness
    float, # speechiness
    str,   # album
    int,   # explicit
    int,   # duration_ms
    int,   # year
    int,   # popularity
    str,   # genre
]
Range = Tuple[float, float]
IntRange = Tuple[int, int]

_SCHEMA_CACHE: Dict[Path, set[str]] = {}


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    resolved = db_path or resolve_db_path()
    return sqlite3.connect(resolved)


def resolve_db_path() -> Path:
    env_raw = os.environ.get("SPOTIFY_DB_PATH")
    if env_raw:
        env_path = Path(env_raw).expanduser().resolve()
        if env_path.exists():
            return env_path

    here = Path(__file__).resolve()
    project_root = here.parents[2] if len(here.parents) >= 3 else here.parent

    candidates = [
        project_root / "data" / "db" / "spotify.db",
        project_root / "data" / "spotify.db",
        project_root / "spotify.db",
        project_root / "src" / "data" / "spotify.db",
        project_root / "src" / "data" / "db" / "spotify.db",
        here.parent / "spotify.db",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find spotify.db. Checked:\n"
        + "\n".join(str(p) for p in candidates)
        + "\n\nSet SPOTIFY_DB_PATH to the full database path if needed."
    )

def has_track_column(*names: str) -> bool:
    columns = _get_schema_columns()
    return _first_existing(columns, *names) is not None

def _get_schema_columns(db_path: Optional[Path] = None) -> set[str]:
    resolved = db_path or resolve_db_path()
    if resolved in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[resolved]

    conn = sqlite3.connect(resolved)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(track)")
        columns = {str(row[1]).lower() for row in cur.fetchall()}
    finally:
        conn.close()

    _SCHEMA_CACHE[resolved] = columns
    return columns


def _first_existing(columns: set[str], *names: str) -> Optional[str]:
    for name in names:
        if name.lower() in columns:
            return name
    return None


def _select_expr(columns: set[str], alias: str, *names: str, default_sql: str) -> str:
    existing = _first_existing(columns, *names)
    if existing is None:
        return f"{default_sql} AS {alias}"
    return f"COALESCE({existing}, {default_sql}) AS {alias}"


def _build_track_select(columns: set[str]) -> str:
    return """
    SELECT
        {id_expr},
        {name_expr},
        {artists_expr},
        {tempo_expr},
        {energy_expr},
        {valence_expr},
        {danceability_expr},
        {acousticness_expr},
        {instrumentalness_expr},
        {liveness_expr},
        {speechiness_expr},
        {album_expr},
        {explicit_expr},
        {duration_expr},
        {year_expr},
        {popularity_expr},
        {genre_expr}
    FROM track
    """.format(
        id_expr=_select_expr(columns, "id", "id", "track_id", default_sql="''"),
        name_expr=_select_expr(columns, "name", "name", "track_name", default_sql="''"),
        artists_expr=_select_expr(columns, "artists", "artists", default_sql="''"),
        tempo_expr=_select_expr(columns, "tempo", "tempo", default_sql="0"),
        energy_expr=_select_expr(columns, "energy", "energy", default_sql="0"),
        valence_expr=_select_expr(columns, "valence", "valence", default_sql="0"),
        danceability_expr=_select_expr(columns, "danceability", "danceability", default_sql="0"),
        acousticness_expr=_select_expr(columns, "acousticness", "acousticness", default_sql="0"),
        instrumentalness_expr=_select_expr(columns, "instrumentalness", "instrumentalness", default_sql="0"),
        liveness_expr=_select_expr(columns, "liveness", "liveness", default_sql="0"),
        speechiness_expr=_select_expr(columns, "speechiness", "speechiness", default_sql="0"),
        album_expr=_select_expr(columns, "album", "album", "album_name", default_sql="''"),
        explicit_expr=_select_expr(columns, "explicit", "explicit", default_sql="0"),
        duration_expr=_select_expr(columns, "duration_ms", "duration_ms", default_sql="0"),
        year_expr=_select_expr(columns, "year", "year", default_sql="0"),
        popularity_expr=_select_expr(columns, "popularity", "popularity", default_sql="0"),
        genre_expr=_select_expr(columns, "genre", "genre", "track_genre", default_sql="''"),
    )


def get_random_tracks(limit: int = 10) -> List[TrackRow]:
    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute(f"{_build_track_select(columns)} ORDER BY RANDOM() LIMIT ?;", (limit,))
        return cur.fetchall()
    finally:
        conn.close()


def get_candidates_for_feature_ranges(
    valence_range: Range,
    energy_range: Range,
    tempo_range: Range,
    danceability_range: Optional[Range] = None,
    acousticness_range: Optional[Range] = None,
    instrumentalness_range: Optional[Range] = None,
    liveness_range: Optional[Range] = None,
    speechiness_range: Optional[Range] = None,
    exclude_artists: Optional[Sequence[str]] = None,
    include_albums: Optional[Sequence[str]] = None,
    exclude_albums: Optional[Sequence[str]] = None,
    include_tracks: Optional[Sequence[str]] = None,
    exclude_tracks: Optional[Sequence[str]] = None,
    include_title_terms: Optional[Sequence[str]] = None,
    exclude_title_terms: Optional[Sequence[str]] = None,
    include_genres: Optional[Sequence[str]] = None,
    exclude_genres: Optional[Sequence[str]] = None,
    include_text_terms: Optional[Sequence[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year: Optional[int] = None,
    year_range: Optional[IntRange] = None,
    decade: Optional[str] = None,
    exclude_explicit: bool = False,
    exclude_live: bool = False,
    exclude_remix: bool = False,
    exclude_christmas: bool = False,
    duration_range_ms: Optional[IntRange] = None,
    limit: int = 500,
) -> List[TrackRow]:
    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    conditions: List[str] = []
    params: List[object] = []

    _add_range_condition(conditions, params, columns, "valence", valence_range)
    _add_range_condition(conditions, params, columns, "energy", energy_range)
    _add_range_condition(conditions, params, columns, "tempo", tempo_range)
    _add_range_condition(conditions, params, columns, "danceability", danceability_range)
    _add_range_condition(conditions, params, columns, "acousticness", acousticness_range)
    _add_range_condition(conditions, params, columns, "instrumentalness", instrumentalness_range)
    _add_range_condition(conditions, params, columns, "liveness", liveness_range)
    _add_range_condition(conditions, params, columns, "speechiness", speechiness_range)

    _add_text_any_condition(conditions, params, _first_existing(columns, "album", "album_name"), include_albums)
    _add_text_none_condition(conditions, params, _first_existing(columns, "album", "album_name"), exclude_albums)

    name_column = _first_existing(columns, "name", "track_name")

    _add_text_any_condition(conditions, params, name_column, include_tracks)
    _add_text_none_condition(conditions, params, name_column, exclude_tracks)

    _add_title_term_any_condition(conditions, params, name_column, include_title_terms)
    _add_title_term_none_condition(conditions, params, name_column, exclude_title_terms)

    _add_text_none_condition(conditions, params, _first_existing(columns, "artists"), exclude_artists)
    _add_genre_any_condition(
        conditions,
        params,
        _first_existing(columns, "genre", "track_genre"),
        include_genres,
    )
    _add_genre_none_condition(
        conditions,
        params,
        _first_existing(columns, "genre", "track_genre"),
        exclude_genres,
    )
    _add_text_any_multi_column_condition(
        conditions,
        params,
        [
            _first_existing(columns, "name", "track_name"),
            _first_existing(columns, "album", "album_name"),
            _first_existing(columns, "artists"),
            _first_existing(columns, "genre", "track_genre"),
        ],
        include_text_terms,
    )

    _add_common_request_filters(
        conditions,
        params,
        columns=columns,
        year=year,
        year_range=year_range,
        decade=decade,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        exclude_explicit=exclude_explicit,
        exclude_live=exclude_live,
        exclude_remix=exclude_remix,
        exclude_christmas=exclude_christmas,
        duration_range_ms=duration_range_ms,
    )

    query = _build_track_select(columns)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY RANDOM() LIMIT ?;"
    params.append(limit)

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
    finally:
        conn.close()


def get_tracks_for_artists(
    artist_names: Sequence[str],
    limit_per_artist: int = 20,
    *,
    valence_range: Optional[Range] = None,
    energy_range: Optional[Range] = None,
    tempo_range: Optional[Range] = None,
    danceability_range: Optional[Range] = None,
    acousticness_range: Optional[Range] = None,
    instrumentalness_range: Optional[Range] = None,
    liveness_range: Optional[Range] = None,
    speechiness_range: Optional[Range] = None,
    include_title_terms: Optional[Sequence[str]] = None,
    exclude_title_terms: Optional[Sequence[str]] = None,
    include_genres: Optional[Sequence[str]] = None,
    exclude_genres: Optional[Sequence[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year: Optional[int] = None,
    year_range: Optional[IntRange] = None,
    decade: Optional[str] = None,
    exclude_explicit: bool = False,
    exclude_live: bool = False,
    exclude_remix: bool = False,
    exclude_christmas: bool = False,
    duration_range_ms: Optional[IntRange] = None,
    ignore_feature_ranges: bool = False,
) -> List[TrackRow]:
    rows: List[TrackRow] = []
    seen_ids = set()

    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    artist_column = _first_existing(columns, "artists")
    genre_column = _first_existing(columns, "genre", "track_genre")
    name_column = _first_existing(columns, "name", "track_name")
    if artist_column is None:
        return rows

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()

        for artist in artist_names:
            cleaned_artist = artist.strip().lower()
            if not cleaned_artist:
                continue

            conditions = [f"LOWER(COALESCE({artist_column}, '')) LIKE ?"]
            params: List[object] = [f"%{cleaned_artist}%"]

            if not ignore_feature_ranges:
                _add_range_condition(conditions, params, columns, "valence", valence_range)
                _add_range_condition(conditions, params, columns, "energy", energy_range)
                _add_range_condition(conditions, params, columns, "tempo", tempo_range)
                _add_range_condition(conditions, params, columns, "danceability", danceability_range)
                _add_range_condition(conditions, params, columns, "acousticness", acousticness_range)
                _add_range_condition(conditions, params, columns, "instrumentalness", instrumentalness_range)
                _add_range_condition(conditions, params, columns, "liveness", liveness_range)
                _add_range_condition(conditions, params, columns, "speechiness", speechiness_range)
            _add_title_term_any_condition(conditions, params, name_column, include_title_terms)
            _add_title_term_none_condition(conditions, params, name_column, exclude_title_terms)
            _add_genre_any_condition(conditions, params, genre_column, include_genres)
            _add_genre_none_condition(conditions, params, genre_column, exclude_genres)
            _add_common_request_filters(
                conditions,
                params,
                columns=columns,
                year=year,
                year_range=year_range,
                decade=decade,
                min_popularity=min_popularity,
                max_popularity=max_popularity,
                exclude_explicit=exclude_explicit,
                exclude_live=exclude_live,
                exclude_remix=exclude_remix,
                exclude_christmas=exclude_christmas,
                duration_range_ms=duration_range_ms,
            )

            query = _build_track_select(columns) + " WHERE " + " AND ".join(conditions) + " ORDER BY RANDOM() LIMIT ?;"
            params.append(limit_per_artist)

            cur.execute(query, params)
            for row in cur.fetchall():
                if not _artist_exact_match(cleaned_artist, row[2]):
                    continue
                if row[0] in seen_ids:
                    continue
                rows.append(row)
                seen_ids.add(row[0])
    finally:
        conn.close()

    return rows


def get_tracks_for_albums(
    album_names: Sequence[str],
    limit_per_album: int = 20,
    *,
    valence_range: Optional[Range] = None,
    energy_range: Optional[Range] = None,
    tempo_range: Optional[Range] = None,
    danceability_range: Optional[Range] = None,
    acousticness_range: Optional[Range] = None,
    instrumentalness_range: Optional[Range] = None,
    liveness_range: Optional[Range] = None,
    speechiness_range: Optional[Range] = None,
    include_title_terms: Optional[Sequence[str]] = None,
    exclude_title_terms: Optional[Sequence[str]] = None,
    include_genres: Optional[Sequence[str]] = None,
    exclude_genres: Optional[Sequence[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year: Optional[int] = None,
    year_range: Optional[IntRange] = None,
    decade: Optional[str] = None,
    exclude_explicit: bool = False,
    exclude_live: bool = False,
    exclude_remix: bool = False,
    exclude_christmas: bool = False,
    duration_range_ms: Optional[IntRange] = None,
    ignore_feature_ranges: bool = False,
) -> List[TrackRow]:
    return _get_tracks_for_named_field(
        field_candidates=["album", "album_name"],
        names=album_names,
        limit_per_name=limit_per_album,
        valence_range=valence_range,
        energy_range=energy_range,
        tempo_range=tempo_range,
        danceability_range=danceability_range,
        acousticness_range=acousticness_range,
        instrumentalness_range=instrumentalness_range,
        liveness_range=liveness_range,
        speechiness_range=speechiness_range,
        include_title_terms=include_title_terms,
        exclude_title_terms=exclude_title_terms,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        year=year,
        year_range=year_range,
        decade=decade,
        exclude_explicit=exclude_explicit,
        exclude_live=exclude_live,
        exclude_remix=exclude_remix,
        exclude_christmas=exclude_christmas,
        duration_range_ms=duration_range_ms,
        ignore_feature_ranges=ignore_feature_ranges,
    )


def get_tracks_by_names(
    track_names: Sequence[str],
    limit_per_name: int = 20,
    *,
    valence_range: Optional[Range] = None,
    energy_range: Optional[Range] = None,
    tempo_range: Optional[Range] = None,
    danceability_range: Optional[Range] = None,
    acousticness_range: Optional[Range] = None,
    instrumentalness_range: Optional[Range] = None,
    liveness_range: Optional[Range] = None,
    speechiness_range: Optional[Range] = None,
    include_title_terms: Optional[Sequence[str]] = None,
    exclude_title_terms: Optional[Sequence[str]] = None,
    include_genres: Optional[Sequence[str]] = None,
    exclude_genres: Optional[Sequence[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year: Optional[int] = None,
    year_range: Optional[IntRange] = None,
    decade: Optional[str] = None,
    exclude_explicit: bool = False,
    exclude_live: bool = False,
    exclude_remix: bool = False,
    exclude_christmas: bool = False,
    duration_range_ms: Optional[IntRange] = None,
    ignore_feature_ranges: bool = False,
) -> List[TrackRow]:
    return _get_tracks_for_named_field(
        field_candidates=["name", "track_name"],
        names=track_names,
        limit_per_name=limit_per_name,
        valence_range=valence_range,
        energy_range=energy_range,
        tempo_range=tempo_range,
        danceability_range=danceability_range,
        acousticness_range=acousticness_range,
        instrumentalness_range=instrumentalness_range,
        liveness_range=liveness_range,
        speechiness_range=speechiness_range,
        include_title_terms=include_title_terms,
        exclude_title_terms=exclude_title_terms,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        year=year,
        year_range=year_range,
        decade=decade,
        exclude_explicit=exclude_explicit,
        exclude_live=exclude_live,
        exclude_remix=exclude_remix,
        exclude_christmas=exclude_christmas,
        duration_range_ms=duration_range_ms,
        ignore_feature_ranges=ignore_feature_ranges,
    )


def _get_tracks_for_named_field(
    *,
    field_candidates: Sequence[str],
    names: Sequence[str],
    limit_per_name: int,
    valence_range: Optional[Range],
    energy_range: Optional[Range],
    tempo_range: Optional[Range],
    danceability_range: Optional[Range],
    acousticness_range: Optional[Range],
    instrumentalness_range: Optional[Range],
    liveness_range: Optional[Range],
    speechiness_range: Optional[Range],
    include_genres: Optional[Sequence[str]],
    exclude_genres: Optional[Sequence[str]],
    min_popularity: Optional[int],
    max_popularity: Optional[int],
    year: Optional[int],
    year_range: Optional[IntRange],
    decade: Optional[str],
    exclude_explicit: bool,
    exclude_live: bool,
    exclude_remix: bool,
    exclude_christmas: bool,
    duration_range_ms: Optional[IntRange],
    ignore_feature_ranges: bool,
    include_title_terms: Optional[Sequence[str]] = None,
    exclude_title_terms: Optional[Sequence[str]] = None,
) -> List[TrackRow]:
    rows: List[TrackRow] = []
    seen_ids = set()

    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    field_name = _first_existing(columns, *field_candidates)
    name_column = _first_existing(columns, "name", "track_name")
    genre_column = _first_existing(columns, "genre", "track_genre")

    if field_name is None:
        return rows

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()

        for name in names:
            cleaned = name.strip().lower()
            if not cleaned:
                continue

            conditions = [f"LOWER(COALESCE({field_name}, '')) LIKE ?"]
            params: List[object] = [f"%{cleaned}%"]

            if not ignore_feature_ranges:
                _add_range_condition(conditions, params, columns, "valence", valence_range)
                _add_range_condition(conditions, params, columns, "energy", energy_range)
                _add_range_condition(conditions, params, columns, "tempo", tempo_range)
                _add_range_condition(conditions, params, columns, "danceability", danceability_range)
                _add_range_condition(conditions, params, columns, "acousticness", acousticness_range)
                _add_range_condition(conditions, params, columns, "instrumentalness", instrumentalness_range)
                _add_range_condition(conditions, params, columns, "liveness", liveness_range)
                _add_range_condition(conditions, params, columns, "speechiness", speechiness_range)

            _add_title_term_any_condition(conditions, params, name_column, include_title_terms)
            _add_title_term_none_condition(conditions, params, name_column, exclude_title_terms)

            _add_genre_any_condition(conditions, params, genre_column, include_genres)
            _add_genre_none_condition(conditions, params, genre_column, exclude_genres)

            _add_common_request_filters(
                conditions,
                params,
                columns=columns,
                year=year,
                year_range=year_range,
                decade=decade,
                min_popularity=min_popularity,
                max_popularity=max_popularity,
                exclude_explicit=exclude_explicit,
                exclude_live=exclude_live,
                exclude_remix=exclude_remix,
                exclude_christmas=exclude_christmas,
                duration_range_ms=duration_range_ms,
            )

            query = _build_track_select(columns) + " WHERE " + " AND ".join(conditions) + " ORDER BY RANDOM() LIMIT ?;"
            params.append(limit_per_name)

            cur.execute(query, params)
            for row in cur.fetchall():
                if row[0] in seen_ids:
                    continue
                rows.append(row)
                seen_ids.add(row[0])
    finally:
        conn.close()

    return rows




def get_tracks_by_ids(track_ids: Sequence[str]) -> List[TrackRow]:
    if not track_ids:
        return []

    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    placeholders = ",".join("?" for _ in track_ids)
    query = _build_track_select(columns) + f" WHERE id IN ({placeholders});"

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute(query, list(track_ids))
        rows = cur.fetchall()
    finally:
        conn.close()

    rows_by_id = {row[0]: row for row in rows}
    return [rows_by_id[track_id] for track_id in track_ids if track_id in rows_by_id]


def _add_common_request_filters(
    conditions: List[str],
    params: List[object],
    *,
    columns: set[str],
    year: Optional[int],
    year_range: Optional[IntRange],
    decade: Optional[str],
    min_popularity: Optional[int],
    max_popularity: Optional[int],
    exclude_explicit: bool,
    exclude_live: bool,
    exclude_remix: bool,
    exclude_christmas: bool,
    duration_range_ms: Optional[IntRange],
) -> None:
    year_column = _first_existing(columns, "year")
    name_column = _first_existing(columns, "name", "track_name")
    album_column = _first_existing(columns, "album", "album_name")
    popularity_column = _first_existing(columns, "popularity")
    explicit_column = _first_existing(columns, "explicit")
    duration_column = _first_existing(columns, "duration_ms")

    if year is not None and year_column is not None:
        conditions.append(f"{year_column} = ?")
        params.append(year)

    resolved_year_range = year_range or _decade_to_year_range(decade)
    if resolved_year_range is not None and year_column is not None:
        conditions.append(f"{year_column} BETWEEN ? AND ?")
        params.extend([resolved_year_range[0], resolved_year_range[1]])

    if min_popularity is not None and popularity_column is not None:
        conditions.append(f"COALESCE({popularity_column}, 0) >= ?")
        params.append(max(0, min(100, min_popularity)))

    if max_popularity is not None and popularity_column is not None:
        conditions.append(f"COALESCE({popularity_column}, 0) <= ?")
        params.append(max(0, min(100, max_popularity)))

    if exclude_explicit and explicit_column is not None:
        conditions.append(f"COALESCE({explicit_column}, 0) = 0")

    if duration_range_ms is not None and duration_column is not None:
        conditions.append(f"COALESCE({duration_column}, 0) BETWEEN ? AND ?")
        params.extend([duration_range_ms[0], duration_range_ms[1]])

    if exclude_live and name_column is not None:
        conditions.append(f"LOWER(COALESCE({name_column}, '')) NOT LIKE ?")
        params.append("%live%")

    if exclude_remix and name_column is not None:
        conditions.append(f"LOWER(COALESCE({name_column}, '')) NOT LIKE ?")
        params.append("%remix%")

    if exclude_christmas and name_column is not None:
        if album_column is None:
            conditions.append(
                f"LOWER(COALESCE({name_column}, '')) NOT LIKE ? AND LOWER(COALESCE({name_column}, '')) NOT LIKE ?"
            )
            params.extend(["%christmas%", "%xmas%"])
        else:
            conditions.append(
                f"LOWER(COALESCE({name_column}, '')) NOT LIKE ? AND "
                f"LOWER(COALESCE({name_column}, '')) NOT LIKE ? AND "
                f"LOWER(COALESCE({album_column}, '')) NOT LIKE ? AND "
                f"LOWER(COALESCE({album_column}, '')) NOT LIKE ?"
            )
            params.extend(["%christmas%", "%xmas%", "%christmas%", "%holiday%"])


def _add_range_condition(
    conditions: List[str],
    params: List[object],
    columns: set[str],
    column_name: str,
    value_range: Optional[Range],
) -> None:
    existing = _first_existing(columns, column_name)
    if value_range is None or existing is None:
        return
    conditions.append(f"{existing} BETWEEN ? AND ?")
    params.extend([value_range[0], value_range[1]])

def _add_genre_any_condition(
    conditions: List[str],
    params: List[object],
    column_name: Optional[str],
    values: Optional[Sequence[str]],
) -> None:
    if column_name is None:
        return

    cleaned = [v.strip().lower() for v in values or [] if v and v.strip()]
    if not cleaned:
        return

    parts = []
    for value in cleaned:
        parts.append(
            f"(',' || REPLACE(REPLACE(LOWER(COALESCE({column_name}, '')), ';', ','), '|', ',') || ',') LIKE ?"
        )
        params.append(f"%,{value},%")

    conditions.append("(" + " OR ".join(parts) + ")")


def _add_genre_none_condition(
    conditions: List[str],
    params: List[object],
    column_name: Optional[str],
    values: Optional[Sequence[str]],
) -> None:
    if column_name is None:
        return

    for value in values or []:
        cleaned = value.strip().lower()
        if not cleaned:
            continue

        conditions.append(
            f"(',' || REPLACE(REPLACE(LOWER(COALESCE({column_name}, '')), ';', ','), '|', ',') || ',') NOT LIKE ?"
        )
        params.append(f"%,{cleaned},%")

def _add_text_any_condition(
    conditions: List[str],
    params: List[object],
    column_name: Optional[str],
    values: Optional[Sequence[str]],
) -> None:
    if column_name is None:
        return
    cleaned = [v.strip().lower() for v in values or [] if v and v.strip()]
    if not cleaned:
        return
    parts = [f"LOWER(COALESCE({column_name}, '')) LIKE ?" for _ in cleaned]
    conditions.append("(" + " OR ".join(parts) + ")")
    params.extend([f"%{v}%" for v in cleaned])


def _add_text_none_condition(
    conditions: List[str],
    params: List[object],
    column_name: Optional[str],
    values: Optional[Sequence[str]],
) -> None:
    if column_name is None:
        return
    for value in values or []:
        cleaned = value.strip().lower()
        if not cleaned:
            continue
        conditions.append(f"LOWER(COALESCE({column_name}, '')) NOT LIKE ?")
        params.append(f"%{cleaned}%")

def _add_text_any_multi_column_condition(
    conditions: List[str],
    params: List[object],
    column_names: Sequence[Optional[str]],
    values: Optional[Sequence[str]],
) -> None:
    columns = [column for column in column_names if column is not None]
    cleaned = [v.strip().lower() for v in values or [] if v and v.strip()]

    if not columns or not cleaned:
        return

    for value in cleaned:
        parts = [f"LOWER(COALESCE({column}, '')) LIKE ?" for column in columns]
        conditions.append("(" + " OR ".join(parts) + ")")
        params.extend([f"%{value}%" for _ in columns])

def _decade_to_year_range(decade: Optional[str]) -> Optional[IntRange]:
    if not decade:
        return None
    dec = decade.strip().lower()
    mapping = {
        "70s": (1970, 1979),
        "80s": (1980, 1989),
        "90s": (1990, 1999),
    }
    if dec in mapping:
        return mapping[dec]
    if dec.endswith("s") and len(dec) == 5 and dec[:4].isdigit():
        start = int(dec[:4])
        return (start, start + 9)
    return None


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


def _artist_exact_match(requested_artist: str, artists_value: object) -> bool:
    requested = requested_artist.strip().lower()
    if not requested:
        return False
    return requested in _split_artist_names(artists_value)

def _split_database_values(value: object) -> List[str]:
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

    parts = re.split(r"\s*(?:;|\||,)\s*", text)
    return [part.strip() for part in parts if part.strip()]


def _normalise_lookup_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def get_available_genres(limit: int = 1000) -> List[str]:
    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    genre_column = _first_existing(columns, "genre", "track_genre")

    if genre_column is None:
        return []

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT DISTINCT {genre_column}
            FROM track
            WHERE TRIM(COALESCE({genre_column}, '')) != ''
            LIMIT ?;
            """,
            (limit,),
        )

        values = set()
        for (raw_value,) in cur.fetchall():
            for value in _split_database_values(raw_value):
                values.add(value.strip())

        return sorted(values, key=str.lower)
    finally:
        conn.close()


def has_genre_match(value: str) -> bool:
    requested = _normalise_lookup_text(value)
    if not requested:
        return True

    available = {
        _normalise_lookup_text(genre)
        for genre in get_available_genres()
    }

    return requested in available


def _has_text_match(field_candidates: Sequence[str], value: str) -> bool:
    requested = value.strip().lower()
    if not requested:
        return True

    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    field_name = _first_existing(columns, *field_candidates)

    if field_name is None:
        return False

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT 1
            FROM track
            WHERE LOWER(COALESCE({field_name}, '')) LIKE ?
            LIMIT 1;
            """,
            (f"%{requested}%",),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def has_track_name_match(value: str) -> bool:
    return _has_text_match(("name", "track_name"), value)


def has_album_match(value: str) -> bool:
    return _has_text_match(("album", "album_name"), value)

def has_artist_match(value: str) -> bool:
    requested = value.strip().lower()
    if not requested:
        return True

    db_path = resolve_db_path()
    columns = _get_schema_columns(db_path)
    artist_column = _first_existing(columns, "artists")

    if artist_column is None:
        return False

    query = (
        _build_track_select(columns)
        + f" WHERE LOWER(COALESCE({artist_column}, '')) LIKE ?"
        + " LIMIT 100;"
    )

    conn = get_connection(db_path)
    try:
        _register_sqlite_functions(conn)
        cur = conn.cursor()
        cur.execute(query, (f"%{requested}%",))
        rows = cur.fetchall()
    finally:
        conn.close()

    return any(_artist_exact_match(requested, row[2]) for row in rows)

def _register_sqlite_functions(conn: sqlite3.Connection) -> None:
    conn.create_function("TITLE_TERM_MATCH", 2, _sqlite_title_term_match)


def _title_term_forms(value: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not cleaned:
        return []

    forms = {cleaned}

    if " " not in cleaned:
        if cleaned.endswith("s") and len(cleaned) > 3:
            forms.add(cleaned[:-1])
        elif len(cleaned) > 2:
            forms.add(cleaned + "s")

    return sorted(forms)


def _sqlite_title_term_match(title: object, term: object) -> int:
    title_text = re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()

    if not title_text:
        return 0

    for form in _title_term_forms(str(term or "")):
        pattern = rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])"
        if re.search(pattern, title_text):
            return 1

    return 0


def _add_title_term_any_condition(
    conditions: List[str],
    params: List[object],
    column_name: Optional[str],
    values: Optional[Sequence[str]],
) -> None:
    if column_name is None:
        return

    cleaned = [v.strip().lower() for v in values or [] if v and v.strip()]
    if not cleaned:
        return

    parts = [f"TITLE_TERM_MATCH({column_name}, ?) = 1" for _ in cleaned]
    conditions.append("(" + " OR ".join(parts) + ")")
    params.extend(cleaned)


def _add_title_term_none_condition(
    conditions: List[str],
    params: List[object],
    column_name: Optional[str],
    values: Optional[Sequence[str]],
) -> None:
    if column_name is None:
        return

    for value in values or []:
        cleaned = value.strip().lower()
        if not cleaned:
            continue

        conditions.append(f"TITLE_TERM_MATCH({column_name}, ?) = 0")
        params.append(cleaned)