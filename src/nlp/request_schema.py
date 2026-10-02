from __future__ import annotations

import math

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# Data model for structured playlist requests.
# This file defines the PlaylistRequest object used across the parser,
# backend, database filtering, ASP fact generation, and playlist builder.


# Shared range types used for audio features, years, and duration values.
Range = Tuple[float, float]
IntRange = Tuple[int, int]


# Represents the structured version of a user's natural-language playlist request.
@dataclass
class PlaylistRequest:
    original_text: str

    emotions: List[str] = field(default_factory=list)
    emotion_weights: Dict[str, float] = field(default_factory=dict)
    activity: Optional[str] = None
    constraint_explanation: Optional[str] = None

    playlist_size: Optional[int] = None
    duration_minutes: Optional[int] = None
    duration_range_ms: Optional[IntRange] = None

    include_artists: List[str] = field(default_factory=list)
    exclude_artists: List[str] = field(default_factory=list)
    artist_target_ratio: Optional[float] = None

    include_albums: List[str] = field(default_factory=list)
    exclude_albums: List[str] = field(default_factory=list)

    include_tracks: List[str] = field(default_factory=list)
    exclude_tracks: List[str] = field(default_factory=list)

    include_title_terms: List[str] = field(default_factory=list)
    exclude_title_terms: List[str] = field(default_factory=list)

    include_genres: List[str] = field(default_factory=list)
    exclude_genres: List[str] = field(default_factory=list)
    min_popularity: Optional[int] = None
    max_popularity: Optional[int] = None

    year: Optional[int] = None
    year_range: Optional[IntRange] = None
    decade: Optional[str] = None

    exclude_explicit: bool = False
    exclude_live: bool = False
    exclude_remix: bool = False
    exclude_christmas: bool = False

    no_repeat_artists: bool = False
    max_tracks_per_artist: Optional[int] = None

    tempo_range: Optional[Range] = None
    energy_range: Optional[Range] = None
    valence_range: Optional[Range] = None
    danceability_range: Optional[Range] = None
    acousticness_range: Optional[Range] = None
    instrumentalness_range: Optional[Range] = None
    liveness_range: Optional[Range] = None
    speechiness_range: Optional[Range] = None

    ordering_style: str = "smooth"


     # Converts the playlist request into a dictionary for JSON responses.
    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


     # Builds a PlaylistRequest from dictionary data returned by the parser or frontend.
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
        original_text_fallback: str = "",
    ) -> "PlaylistRequest":
        def _float_range(name: str) -> Optional[Range]:
            value = data.get(name)
            if value in (None, "", []):
                return None
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name} must be a two-item list or tuple")
            return float(value[0]), float(value[1])

        def _int_range(name: str) -> Optional[IntRange]:
            value = data.get(name)
            if value in (None, "", []):
                return None
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name} must be a two-item list or tuple")
            return int(value[0]), int(value[1])

        def _int_in_range(name: str) -> Optional[int]:
            value = _optional_int(data.get(name))
            if value is None:
                return None
            return max(0, min(100, value))

        return cls(
            original_text=str(data.get("original_text") or original_text_fallback),
            emotions=[str(x) for x in data.get("emotions", []) or []],
            emotion_weights={
                str(k): float(v)
                for k, v in (data.get("emotion_weights", {}) or {}).items()
            },
            activity=_optional_str(data.get("activity")),
            constraint_explanation=_optional_str(data.get("constraint_explanation")),
            playlist_size=_optional_int(data.get("playlist_size")),
            duration_minutes=_optional_int(data.get("duration_minutes")),
            duration_range_ms=_int_range("duration_range_ms"),
            include_artists=[str(x) for x in data.get("include_artists", []) or []],
            exclude_artists=[str(x) for x in data.get("exclude_artists", []) or []],
            artist_target_ratio=_optional_float_ratio(data.get("artist_target_ratio")),
            include_albums=[str(x) for x in data.get("include_albums", []) or []],
            exclude_albums=[str(x) for x in data.get("exclude_albums", []) or []],
            include_tracks=[str(x) for x in data.get("include_tracks", []) or []],
            exclude_tracks=[str(x) for x in data.get("exclude_tracks", []) or []],
            include_title_terms=[str(x) for x in data.get("include_title_terms", []) or []],
            exclude_title_terms=[str(x) for x in data.get("exclude_title_terms", []) or []],
            include_genres=[str(x) for x in data.get("include_genres", []) or []],
            exclude_genres=[str(x) for x in data.get("exclude_genres", []) or []],
            min_popularity=_int_in_range("min_popularity"),
            max_popularity=_int_in_range("max_popularity"),
            year=_optional_int(data.get("year")),
            year_range=_int_range("year_range"),
            decade=_optional_str(data.get("decade")),
            exclude_explicit=bool(data.get("exclude_explicit", False)),
            exclude_live=bool(data.get("exclude_live", False)),
            exclude_remix=bool(data.get("exclude_remix", False)),
            exclude_christmas=bool(data.get("exclude_christmas", False)),
            no_repeat_artists=bool(data.get("no_repeat_artists", False)),
            max_tracks_per_artist=_optional_int(data.get("max_tracks_per_artist")),
            tempo_range=_float_range("tempo_range"),
            energy_range=_float_range("energy_range"),
            valence_range=_float_range("valence_range"),
            danceability_range=_float_range("danceability_range"),
            acousticness_range=_float_range("acousticness_range"),
            instrumentalness_range=_float_range("instrumentalness_range"),
            liveness_range=_float_range("liveness_range"),
            speechiness_range=_float_range("speechiness_range"),
            ordering_style=str(data.get("ordering_style") or "smooth"),
        )

     # Resolves the final number of tracks to generate.
    def resolved_track_count(self, default: int = 10) -> int:
        if self.playlist_size is not None and self.playlist_size > 0:
            return self.playlist_size

        if self.duration_minutes is not None and self.duration_minutes > 0:
            estimated = round(self.duration_minutes / 3.5)
            return max(3, estimated)

        return default

    # Calculates the minimum spacing used when the user asks for no repeated artists.
    def resolved_artist_gap(self, track_count: int) -> int:
        if not self.no_repeat_artists:
            return 0
        return max(2, (track_count + 4) // 5)
    
    # Calculates how many tracks should come from explicitly requested artists.
    def resolved_artist_target_count(self, track_count: int) -> int:
        if not self.include_artists:
            return 0

        ratio = self.artist_target_ratio
        if ratio is None:
            ratio = 0.10

        return max(1, min(track_count, int(math.ceil(track_count * ratio))))


# Converts empty values into None and keeps non-empty strings.
def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

# Converts an optional value into an integer.
def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)

# Converts an optional value into a ratio between 0.0 and 1.0.
def _optional_float_ratio(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return max(0.0, min(1.0, float(value)))

