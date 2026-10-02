from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Tuple

from .emotions import EMOTION_PROFILES, known_emotion_tokens, text_to_emotion_weights
from .request_schema import PlaylistRequest

# Natural-language request parser for IntelliPlaylist.
# This file uses an LLM parser when available and falls back to rule-based parsing.
# It converts a user's playlist prompt into a structured PlaylistRequest object.


# Maximum allowed track duration used when interpreting long-song requests.
MAX_TRACK_DURATION_MS = 20 * 60 * 1000

# System prompt used to guide the LLM into returning a structured playlist request.
LLM_SYSTEM_PROMPT = """
You convert playlist requests from natural language into structured JSON.

Return JSON only with these keys:
- original_text
- emotions
- emotion_weights
- activity
- constraint_explanation
- playlist_size
- duration_minutes
- duration_range_ms
- include_artists
- exclude_artists
- artist_target_ratio
- include_albums
- exclude_albums
- include_tracks
- exclude_tracks
- include_title_terms
- exclude_title_terms
- include_genres
- exclude_genres
- min_popularity
- max_popularity
- year
- year_range
- decade
- exclude_explicit
- exclude_remix
- exclude_live
- exclude_christmas
- no_repeat_artists
- max_tracks_per_artist
- tempo_range
- energy_range
- valence_range
- danceability_range
- acousticness_range
- instrumentalness_range
- liveness_range
- speechiness_range
- ordering_style

Canonical emotion profiles:
- joy
- energetic
- calm
- focus
- sad
- anger
- romantic
- nostalgic
- confident
- dark
- dreamy
- hopeful
- tense
- sensual

Rules:
- Keep original_text exactly as given.
- Only fill values that are explicit or strongly implied by the user.
- Use null for missing single values.
- Use [] for missing lists.
- Use {} for missing emotion_weights.
- Do not invent artists, albums, tracks, genres, years, durations, or ranges.

Emotion interpretation:
- emotion_weights must only use these keys:
  joy, energetic, calm, focus, sad, anger, romantic, nostalgic, confident, dark, dreamy, hopeful, tense, sensual.
- The values in emotion_weights should sum to 1.0 when present.
- Use emotion_weights to represent the emotional meaning of activity, mood, and vibe words.
- If the user says words like gym, workout, party, study, relaxing, chill, fun, aggressive, romantic, nostalgic, confident, dark, dreamy, hopeful, tense, or sensual, infer a reasonable weighting over the canonical emotion profiles.
- Keep the raw user words in emotions when useful.
- If the request is neutral and only asks for playlist size, duration, artist constraints, album constraints, track constraints, genre, popularity, year, or exclusions, then set:
  - emotions = []
  - emotion_weights = {}
  - activity = null

Artist interpretation:
- Use include_artists when the user asks to include, add, feature, or have music by an artist.
- artist_target_ratio represents the approximate share of the final playlist that should come from the requested included artists as a group.
- If the user includes one or more artists with no quantity word, set artist_target_ratio = 0.10.
- If the user says "some", "a few", or "add some" for included artists, set artist_target_ratio = 0.25.
- If the user says "a lot", "lots of", "loads of", "mostly", or "heavy on" for included artists, set artist_target_ratio = 0.50.
- If the user says "only", "all", or "just" for included artists, set artist_target_ratio = 1.00.
- If no artist is included, set artist_target_ratio = null.
- Do not infer emotions, activity, ordering_style, or audio-feature ranges from artist names alone.

Title/theme interpretation:
- Use include_title_terms when the user asks for songs "about", "called", "with title", "with the word", or "containing" a topic/word in the song title.
- These are title keywords, not exact track names.
- Do not put title/theme keywords into include_tracks unless the user clearly names a specific song.
- Examples:
  - "a song about dogs" -> include_title_terms = ["dog"]
  - "songs about rain" -> include_title_terms = ["rain"]
  - "tracks with love in the title" -> include_title_terms = ["love"]
  - "no songs about Christmas" -> exclude_title_terms = ["christmas"]
  - "include the song Time by Pink Floyd" -> include_tracks = ["Time"], include_title_terms = []

Genre, popularity, and year interpretation:
- Use include_genres / exclude_genres only when the user clearly names genres.
- Use min_popularity / max_popularity only for clear popularity requests.
- "popular", "mainstream", "well-known", "familiar", "recognisable", "classic hits", "songs everyone knows" -> min_popularity around 60 or 70.
- "very popular", "big hits", "famous songs" -> min_popularity around 75 or 80.
- "obscure", "underground", "lesser-known", "deep cuts" -> max_popularity around 40.
- Do not interpret "loud", "quiet", "gentle", or "soft" as popularity.
- "nothing too loud", "not too loud", "quiet", "gentle", "soft" should affect energy_range, not max_popularity.
- Use year, year_range, or decade only when the user clearly asks for a year, year range, or decade.

Ordering interpretation:
- If the user asks for smooth transitions, use ordering_style = "smooth".
- If the user says start calm then build up, start slow then build, or gradually build, use ordering_style = "build_up".
- If the user says high energy throughout, keep the energy high, or energetic throughout, use ordering_style = "high_energy".
- If the user says don't repeat artists, no repeat artists, one song per artist, or no artist repeats, set no_repeat_artists = true.

Duration interpretation:
- Duration wording must distinguish between total playlist duration and individual song/track duration.
- If the adjective describes "playlist", "mix", "set", or "music for X minutes", use duration_minutes.
- If the adjective describes "songs" or "tracks", use duration_range_ms.
- If the request contains both total playlist duration and individual song duration, fill both fields separately.
- Do not let "long playlist" imply long songs.
- Do not let "short songs" imply a short playlist.

Playlist duration examples:
- "short playlist", "quick playlist", "brief playlist" -> duration_minutes less than 10 songs
- "quite a long playlist", "long playlist", "long mix" -> more than 25 songs
- "very long playlist", "huge playlist", "massive playlist" -> a lot more than 25 songs
- "a 30 minute playlist", "an hour long playlist", "music for a 45 minute workout" -> use the explicit total duration.

Track duration examples:
- "short songs", "quick songs", "brief tracks" -> duration_range_ms around [0, 180000].
- "quite short songs" -> duration_range_ms around [90000, 210000].
- "medium length songs", "normal length songs" -> duration_range_ms around [180000, 300000].
- "quite long songs", "longer songs" -> duration_range_ms around [240000, 480000].
- "long songs", "extended tracks" -> duration_range_ms around [300000, 900000].
- "very long songs", "epic length tracks" -> duration_range_ms around [420000, 1200000].
- "a long playlist with short songs" -> more than 25 songs and duration_range_ms around [0, 180000].
- "a short playlist with long songs" -> less than 10 songs and duration_range_ms around [300000, 900000].

Audio feature interpretation:
- Audio feature ranges should only be filled when the user explicitly asks for that sound quality or strongly implies it.
- Do not fill audio feature ranges just because the user gives a general mood, unless the connection is strong and useful.

Important speechiness/instrumentalness distinction:
- speechiness_range measures spoken, speech-like, talky, rap-heavy, or spoken-word audio.
- instrumentalness_range measures whether a track is likely to have no vocals.
- Do not treat speechiness_range as "amount of lyrics" or "amount of words" in normal sung music.
- Do not set speechiness_range just because a song has normal vocals or lyrics.
- Important: "no words", "no lyrics", and "no vocals" do NOT mean low speechiness_range. They mean the user wants vocal-free / instrumental music, so use instrumentalness_range instead.

Vocal-free and instrumental requests:
- "no words", "no lyrics", "no vocals", "wordless", "vocal-free", "without vocals", "without lyrics" -> instrumentalness_range around [0.7, 1.0] and speechiness_range = null.
- "instrumental", "instrumental only", "only instrumental" -> instrumentalness_range around [0.7, 1.0] and speechiness_range = null.
- "strictly instrumental", "completely wordless", "absolutely no vocals" -> instrumentalness_range around [0.9, 1.0] and speechiness_range = null.
- "mostly instrumental", "minimal vocals", "not many vocals" -> instrumentalness_range around [0.5, 1.0] and speechiness_range = null.
- In these cases, leave speechiness_range as null unless the user also mentions speech, talking, rap, or spoken-word.

Speech-like requests:
- "spoken word", "speech-heavy", "talking", "talky", "podcast-like", "mostly spoken" -> speechiness_range around [0.45, 1.0].
- "rap-heavy", "lots of rap", "bars", "lyrical rap" -> speechiness_range around [0.25, 0.6].
- "no talking", "no speech", "no spoken word", "nothing spoken", "not speechy" -> speechiness_range = [0.0, 0.0].

General audio feature examples:
- "fast", "high tempo", "quick pace" -> tempo_range around [120, 170].
- "very fast", "rapid", "speedy" -> tempo_range around [140, 190].
- "slow", "low tempo", "laid back pace" -> tempo_range around [60, 100].
- "very slow" -> tempo_range around [50, 85].
- "high energy", "intense", "hype", "pumped" -> energy_range around [0.75, 1.0].
- "very high energy", "maximum energy" -> energy_range around [0.85, 1.0].
- "low energy", "gentle", "soft" -> energy_range around [0.0, 0.35].
- "very low energy", "sleepy", "quiet energy" -> energy_range around [0.0, 0.2].
- "danceable", "groovy", "bouncy", "dancey" -> danceability_range around [0.65, 1.0].
- "very danceable", "clubby" -> danceability_range around [0.8, 1.0].
- "acoustic", "stripped back", "unplugged" -> acousticness_range around [0.6, 1.0].
- "very acoustic", "fully acoustic" -> acousticness_range around [0.8, 1.0].
- "live sounding", "live feel", "concert feel" -> liveness_range around [0.5, 1.0].
- "positive", "feel-good", "happy", "upbeat" -> valence_range around [0.65, 1.0].
- "very happy", "very positive" -> valence_range around [0.8, 1.0].
- "sad", "melancholy", "dark mood", "low valence" -> valence_range around [0.0, 0.35].
- "not too loud", "nothing too loud", "not loud", "quiet", "gentle", "soft" -> energy_range around [0.0, 0.45].

Absolute negative audio feature wording:
- For audio features measured from 0.0 to 1.0, interpret absolute negative wording as zero only when the wording clearly maps to that audio feature.
- Absolute negative wording includes "no", "zero", "nothing", "none", "without", and "-free".
- If the user uses absolute negative wording directly before a 0.0-to-1.0 audio feature, set that feature's range to [0.0, 0.0].
- Do not apply this blindly to natural-language concepts that are not the same as the audio feature.
- Do not apply the exact-zero audio feature rule to tempo, duration, playlist size, artists, albums, tracks, genres, or general mood words.

Exact zero examples:
- "no acoustic", "nothing acoustic", "acoustic-free" -> acousticness_range = [0.0, 0.0].
- "no instrumental parts", "nothing instrumental" -> instrumentalness_range = [0.0, 0.0].
- "no live feel", "nothing live sounding" -> liveness_range = [0.0, 0.0].
- "not danceable", "nothing dancey", "no danceability" -> danceability_range = [0.0, 0.0].
- "no energy", "zero energy" -> energy_range = [0.0, 0.0].
- "no happiness", "nothing upbeat", "not positive" -> valence_range = [0.0, 0.0].

Softer negative audio wording:
- For softer negative wording, use a low range instead of exact zero.
- Softer negative wording includes "low", "not much", "barely any", "minimal", "a little", "not too", and "less".
- "low energy", "not much energy", "gentle", "soft" -> energy_range around [0.0, 0.35].
- "barely acoustic", "not too acoustic", "low acousticness" -> acousticness_range around [0.0, 0.25].
- "minimal live feel", "not live sounding" -> liveness_range around [0.0, 0.25].
- "low danceability", "not very danceable" -> danceability_range around [0.0, 0.4].
- "not many vocals", "minimal vocals" -> instrumentalness_range around [0.5, 1.0].

Exclusion-style requests:
- Use the relevant exclusion field instead of an audio-feature range when the user is excluding a category of tracks.
- "no live songs", "no live versions" -> exclude_live = true.
- "no remixes", "avoid remixes" -> exclude_remix = true.
- "no explicit songs", "clean only" -> exclude_explicit = true.
- "no Christmas songs", "avoid Christmas" -> exclude_christmas = true.

constraint_explanation:
- constraint_explanation should briefly explain how the request was interpreted.
- Keep constraint_explanation to 1-3 short sentences.
- Mention the most important explicit constraints and any strong inference such as activity, duration, popularity, artist quantity, or audio features.
- Do not mention internal implementation details.
""".strip()


# Activity keywords used by the rule-based fallback parser.
ACTIVITY_KEYWORDS = {
    "workout": ["gym", "workout", "run", "running", "exercise", "training"],
    "study": ["study", "focus", "revision", "reading"],
    "party": ["party", "pregame", "pre drinks"],
    "sleep": ["sleep", "bedtime"],
    "relax": ["relax", "relaxing", "wind down", "unwind"],
}

# Ordering keywords used to detect the requested playlist flow.
ORDERING_KEYWORDS = {
    "build_up": [
        "build up",
        "start calm then build",
        "start slow then build",
        "gradually build",
    ],
    "high_energy": [
        "high energy throughout",
        "keep the energy high",
        "energetic throughout",
    ],
    "cool_down": [
        "cool down",
        "calm ending",
        "end calmer",
        "finish calm",
    ],
    "smooth": [
        "smooth transitions",
        "smooth flow",
        "smooth order",
    ],
}


# Parses a natural-language playlist request using the LLM first,
# then falls back to rule-based parsing if the LLM cannot be used.
def parse_playlist_request(
    user_text: str,
    *,
    llm_callable: Optional[Callable[[str, str], str]] = None,
    debug: bool = False,
) -> PlaylistRequest:
    if llm_callable is not None:
        try:
            raw = llm_callable(LLM_SYSTEM_PROMPT, user_text)
            request = _request_from_json(raw, user_text)
            request = _strip_emotion_fields_if_neutral(request)
            request = _normalise_emotion_fields(request)
            if debug:
                print("[PARSER] Used LLM parser.")
            return request
        except Exception as exc:
            if debug:
                print(f"[PARSER] LLM parsing failed, falling back to rules: {exc}")

    request = _rule_based_parse(user_text, debug=debug)
    request = _strip_emotion_fields_if_neutral(request)
    request = _normalise_emotion_fields(request)
    if debug:
        print("[PARSER] Used rule-based parser.")
    return request


# Converts raw LLM JSON output into a PlaylistRequest object.
def _request_from_json(raw_json: str, original_text: str) -> PlaylistRequest:
    data = json.loads(_extract_json_object(raw_json))
    return PlaylistRequest.from_dict(data, original_text_fallback=original_text)

# Extracts the JSON object from the LLM response text.
def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return text[start:end + 1]

# Rule-based fallback parser used when LLM parsing fails or is unavailable.
def _rule_based_parse(user_text: str, *, debug: bool = False) -> PlaylistRequest:
    lower = user_text.lower()

    emotions = [
        token
        for token in known_emotion_tokens()
        if re.search(rf"\b{re.escape(token)}\b", lower)
    ]

    emotion_weights = text_to_emotion_weights(user_text, debug=debug)

    activity = _extract_activity(lower)
    max_tracks_per_artist = _extract_max_tracks_per_artist(lower)
    playlist_size = _extract_playlist_size(lower, max_tracks_per_artist=max_tracks_per_artist)
    duration_minutes = _extract_playlist_duration_minutes(lower)
    duration_range_ms = _extract_track_duration_range_ms(lower)

    include_albums = _extract_named_items(
        user_text,
        entity_keywords=["album", "record", "ep"],
        action="include",
    )
    exclude_albums = _extract_named_items(
        user_text,
        entity_keywords=["album", "record", "ep"],
        action="exclude",
    )

    include_tracks = _extract_named_items(
        user_text,
        entity_keywords=["song", "track"],
        action="include",
    )
    exclude_tracks = _extract_named_items(
        user_text,
        entity_keywords=["song", "track"],
        action="exclude",
    )

    include_genres = _extract_named_items(
        user_text,
        entity_keywords=["genre", "genres"],
        action="include",
    )
    exclude_genres = _extract_named_items(
        user_text,
        entity_keywords=["genre", "genres"],
        action="exclude",
    )

    include_artists = _extract_include_artists(
        user_text,
        already_used=include_albums + include_tracks + include_genres,
    )
    exclude_artists = _extract_exclude_artists(
        user_text,
        already_used=exclude_albums + exclude_tracks + exclude_genres,
    )
    artist_target_ratio = _extract_artist_target_ratio(lower, include_artists)

    min_popularity, max_popularity = _extract_popularity_range(lower)
    year, year_range, decade = _extract_year_info(lower)

    exclude_explicit = bool(
        "no explicit" in lower
        or "no explicit songs" in lower
        or "clean only" in lower
    )
    exclude_remix = bool(
        "avoid remixes" in lower
        or "no remixes" in lower
        or "no remix" in lower
        or "exclude remixes" in lower
    )
    exclude_live = bool(
        "avoid live" in lower
        or "no live" in lower
        or "exclude live" in lower
        or "no live versions" in lower
    )
    exclude_christmas = bool(
        "no christmas" in lower
        or "avoid christmas" in lower
        or "exclude christmas" in lower
        or "no christmas songs" in lower
    )

    no_repeat_artists = bool(
        re.search(r"\b(don'?t|do not|no) repeat artists\b", lower)
        or "no artist repeats" in lower
        or "one song per artist" in lower
    )

    ordering_style = _extract_ordering_style(lower)

    (
        tempo_range,
        energy_range,
        valence_range,
        danceability_range,
        acousticness_range,
        instrumentalness_range,
        liveness_range,
        speechiness_range,
    ) = _extract_audio_ranges(lower)

    constraint_explanation = _build_constraint_explanation(
        include_artists=include_artists,
        exclude_artists=exclude_artists,
        include_albums=include_albums,
        include_tracks=include_tracks,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        playlist_size=playlist_size,
        duration_minutes=duration_minutes,
        activity=activity,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        ordering_style=ordering_style,
    )

    return PlaylistRequest(
        original_text=user_text,
        emotions=emotions,
        emotion_weights=emotion_weights,
        activity=activity,
        constraint_explanation=constraint_explanation,
        playlist_size=playlist_size,
        duration_minutes=duration_minutes,
        duration_range_ms=duration_range_ms,
        include_artists=include_artists,
        exclude_artists=exclude_artists,
        artist_target_ratio=artist_target_ratio,
        include_albums=include_albums,
        exclude_albums=exclude_albums,
        include_tracks=include_tracks,
        exclude_tracks=exclude_tracks,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        min_popularity=min_popularity,
        max_popularity=max_popularity,
        year=year,
        year_range=year_range,
        decade=decade,
        exclude_explicit=exclude_explicit,
        exclude_remix=exclude_remix,
        exclude_live=exclude_live,
        exclude_christmas=exclude_christmas,
        no_repeat_artists=no_repeat_artists,
        max_tracks_per_artist=max_tracks_per_artist,
        tempo_range=tempo_range,
        energy_range=energy_range,
        valence_range=valence_range,
        danceability_range=danceability_range,
        acousticness_range=acousticness_range,
        instrumentalness_range=instrumentalness_range,
        liveness_range=liveness_range,
        speechiness_range=speechiness_range,
        ordering_style=ordering_style,
    )


# Builds a short explanation of the main constraints detected in the request.
def _build_constraint_explanation(
    *,
    include_artists: List[str],
    exclude_artists: List[str],
    include_albums: List[str],
    include_tracks: List[str],
    include_genres: List[str],
    exclude_genres: List[str],
    playlist_size: Optional[int],
    duration_minutes: Optional[int],
    activity: Optional[str],
    min_popularity: Optional[int],
    max_popularity: Optional[int],
    ordering_style: str,
) -> str:
    parts: List[str] = []

    if include_artists:
        parts.append(f"Included artist preferences: {', '.join(include_artists)}.")
    if exclude_artists:
        parts.append(f"Excluded artists: {', '.join(exclude_artists)}.")
    if include_albums:
        parts.append(f"Included albums: {', '.join(include_albums)}.")
    if include_tracks:
        parts.append(f"Included tracks: {', '.join(include_tracks)}.")
    if include_genres:
        parts.append(f"Included genres: {', '.join(include_genres)}.")
    if exclude_genres:
        parts.append(f"Excluded genres: {', '.join(exclude_genres)}.")
    if playlist_size is not None:
        parts.append(f"Requested playlist size: {playlist_size} tracks.")
    if duration_minutes is not None:
        parts.append(f"Target playlist duration: {duration_minutes} minutes.")
    if activity:
        parts.append(f"Detected activity or vibe: {activity}.")
    if min_popularity is not None:
        parts.append(f"Applied a minimum popularity of {min_popularity}.")
    if max_popularity is not None:
        parts.append(f"Applied a maximum popularity of {max_popularity}.")
    if ordering_style != "smooth":
        parts.append(f"Used an ordering style of {ordering_style}.")

    if not parts:
        return "Interpreted the request using only the explicitly stated playlist details."

    return " ".join(parts[:3])


# Extracts the activity or use-case from the user's request.
def _extract_activity(lower: str) -> Optional[str]:
    for label, keywords in ACTIVITY_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return label
    return None

# Extracts the requested playlist size from phrases such as "10 songs".
def _extract_playlist_size(
    lower: str,
    *,
    max_tracks_per_artist: Optional[int] = None,
) -> Optional[int]:
    strong_patterns = [
        r"\bplaylist of\s+(\d+)\s*(songs?|tracks?)\b",
        r"\bwith\s+(\d+)\s*(songs?|tracks?)\b",
        r"\bmake(?: me)?\s+(?:a\s+)?playlist\s+of\s+(\d+)\s*(songs?|tracks?)\b",
        r"\bgive me\s+(\d+)\s*(songs?|tracks?)\b",
        r"\bwant\s+(\d+)\s*(songs?|tracks?)\b",
        r"\bneed\s+(\d+)\s*(songs?|tracks?)\b",
        r"\bcreate\s+(\d+)\s*(songs?|tracks?)\b",
    ]

    for pattern in strong_patterns:
        match = re.search(pattern, lower, flags=re.VERBOSE)
        if match:
            return int(match.group(1))

    match = re.search(r"\b(\d+)\s*(songs?|tracks?)\b(?!\s+per artist)", lower)
    if match:
        value = int(match.group(1))
        if max_tracks_per_artist is not None and value == max_tracks_per_artist:
            per_artist_phrase = re.search(
                rf"\b(?:max(?:imum)?|at most)?\s*{value}\s*(songs?|tracks?)\s+per artist\b",
                lower,
            )
            if per_artist_phrase:
                return None
        return value

    return None


# Extracts the total requested playlist duration in minutes.
def _extract_playlist_duration_minutes(lower: str) -> Optional[int]:
    if re.search(r"\bhalf\s+(?:an?\s+)?hour\b", lower):
        return 30
    if re.search(r"\ban?\s+hour\b", lower):
        return 60
    if re.search(r"\bone\s+hour\b", lower):
        return 60

    hour_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(hours?|hr|hrs)\b", lower)
    if hour_match:
        return int(round(float(hour_match.group(1)) * 60))

    if re.search(r"\ban?\s+minute\b", lower):
        return 1
    if re.search(r"\bone\s+minute\b", lower):
        return 1

    minute_match = re.search(r"\b(\d+)\s*(minutes?|min|mins)\b", lower)
    if minute_match:
        return int(minute_match.group(1))

    if "enough for a workout" in lower:
        return 45

    return None


# Extracts requested individual track duration ranges in milliseconds.
def _extract_track_duration_range_ms(lower: str) -> Optional[Tuple[int, int]]:
    between_match = re.search(
        r"\b(?:songs?|tracks?)\s+(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*(\d+(?:\.\d+)?)\s*minutes?\b",
        lower,
    )
    if between_match:
        low = _minutes_to_ms(float(between_match.group(1)))
        high = _minutes_to_ms(float(between_match.group(2)))
        return (min(low, high), max(low, high))

    under_match = re.search(
        r"\b(?:songs?|tracks?)\s+(?:under|below|less than)\s+(\d+(?:\.\d+)?)\s*minutes?\b",
        lower,
    )
    if under_match:
        high = _minutes_to_ms(float(under_match.group(1)))
        return (0, high)

    over_match = re.search(
        r"\b(?:songs?|tracks?)\s+(?:over|above|more than)\s+(\d+(?:\.\d+)?)\s*minutes?\b",
        lower,
    )
    if over_match:
        low = _minutes_to_ms(float(over_match.group(1)))
        return (low, MAX_TRACK_DURATION_MS)

    compact_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*minute\s+(?:songs?|tracks?)\b",
        lower,
    )
    if compact_match:
        low = _minutes_to_ms(float(compact_match.group(1)))
        high = _minutes_to_ms(float(compact_match.group(2)))
        return (min(low, high), max(low, high))

    return None


# Converts a duration in minutes into milliseconds.
def _minutes_to_ms(minutes: float) -> int:
    return int(round(minutes * 60_000))

# Extracts named albums, tracks, or genres from include/exclude phrases.
def _extract_named_items(
    text: str,
    *,
    entity_keywords: List[str],
    action: str,
) -> List[str]:
    if action not in {"include", "exclude"}:
        raise ValueError("action must be 'include' or 'exclude'")

    if action == "include":
        verbs = ["include", "add", "with", "must include"]
    else:
        verbs = ["exclude", "no", "avoid", "without"]

    items: List[str] = []

    for verb in verbs:
        for entity in entity_keywords:
            pattern = (
                rf"(?:^|[,.]\s*|\band\s+)"
                rf"{re.escape(verb)}\s+"
                rf"(?:the\s+)?{re.escape(entity)}\s+"
                rf'["“]?([^,.;"”]+)["”]?'
            )
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = match.group(1).strip()
                if candidate:
                    items.extend(_split_multi_value_phrase(candidate))

    return _dedupe_preserve_order(items)


# Splits phrases containing multiple values into separate items.
def _split_multi_value_phrase(text: str) -> List[str]:
    raw_parts = re.split(r"\s*(?:,|/| and )\s*", text, flags=re.IGNORECASE)
    cleaned = []
    for part in raw_parts:
        value = part.strip().strip('"').strip("'")
        if value:
            cleaned.append(value)
    return cleaned


# Extracts artist names that the user wants included.
def _extract_include_artists(text: str, already_used: List[str]) -> List[str]:
    phrases = _extract_after_keywords(
        text,
        ["include", "add some", "add", "lots of", "more of", "with", "must include"],
    )
    return _filter_possible_artists(phrases, already_used=already_used)

# Estimates how much of the playlist should come from requested artists.
def _extract_artist_target_ratio(
    lower: str,
    include_artists: List[str],
) -> Optional[float]:
    if not include_artists:
        return None

    if re.search(r"\b(a lot of|lots of|loads of)\b", lower):
        return 0.50

    if re.search(r"\b(some|a few|add some|bit of)\b", lower):
        return 0.50

    return 0.10

# Extracts artist names that the user wants excluded.
def _extract_exclude_artists(text: str, already_used: List[str]) -> List[str]:
    generic_phrases = [
        "explicit",
        "explicit songs",
        "remix",
        "remixes",
        "live",
        "live versions",
        "christmas",
        "christmas songs",
        "repeat artists",
        "artist repeats",
        "genre",
        "genres",
    ]

    phrases = _extract_after_keywords(text, ["no", "avoid", "without", "exclude"])
    filtered: List[str] = []
    for phrase in phrases:
        candidate = phrase.strip().lower()
        if not candidate:
            continue
        if any(candidate.startswith(x) for x in generic_phrases):
            continue
        filtered.append(phrase)

    return _filter_possible_artists(filtered, already_used=already_used)


# Extracts text following keywords such as include, avoid, or without.
def _extract_after_keywords(text: str, keywords: List[str]) -> List[str]:
    results: List[str] = []
    for keyword in keywords:
        pattern = rf"(?:^|[,.]\s*|\band\s+){re.escape(keyword)}\s+([^,.;]+)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = match.group(1).strip()
            phrase = re.split(
                r"\b(?:playlist|songs?|tracks?|for|please|but)\b",
                phrase,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            phrase = re.sub(
                r"^(?:some|more|lots of|the)\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            ).strip()
            if phrase:
                results.append(phrase)
    return _dedupe_preserve_order(results)


# Filters extracted phrases down to likely artist names.
def _filter_possible_artists(
    items: List[str],
    *,
    already_used: List[str],
) -> List[str]:
    blocked_tokens = {
        "explicit",
        "explicit songs",
        "remix",
        "remixes",
        "live",
        "live versions",
        "christmas",
        "christmas songs",
        "album",
        "record",
        "ep",
        "song",
        "track",
        "genre",
        "genres",
        "playlist",
        "songs",
        "tracks",
    }

    used_lower = {x.lower() for x in already_used}
    filtered: List[str] = []

    for item in items:
        candidate = item.strip().strip('"').strip("'")
        if not candidate:
            continue

        candidate_lower = candidate.lower()

        if candidate_lower in blocked_tokens:
            continue
        if candidate_lower.startswith("album "):
            continue
        if candidate_lower.startswith("song "):
            continue
        if candidate_lower.startswith("track "):
            continue
        if candidate_lower.startswith("genre "):
            continue
        if candidate_lower in used_lower:
            continue

        filtered.append(candidate.title())

    return _dedupe_preserve_order(filtered)

# Extracts popularity constraints from wording such as popular, obscure, or numeric thresholds.
def _extract_popularity_range(lower: str) -> Tuple[Optional[int], Optional[int]]:
    if any(token in lower for token in [
        "very high popularity",
        "extremely popular",
        "very popular",
        "super popular",
    ]):
        return 80, None

    if any(token in lower for token in [
        "high popularity",
        "highly popular",
    ]):
        return 70, None

    between = re.search(r"\bpopularity\s*(?:between|from)\s*(\d{1,3})\s*(?:and|to|-)\s*(\d{1,3})\b", lower)
    if between:
        low = max(0, min(100, int(between.group(1))))
        high = max(0, min(100, int(between.group(2))))
        return min(low, high), max(low, high)

    above = re.search(r"\bpopularity\s*(?:above|over|at least|>=?)\s*(\d{1,3})\b", lower)
    if above:
        return max(0, min(100, int(above.group(1)))), None

    below = re.search(r"\bpopularity\s*(?:below|under|at most|<=?)\s*(\d{1,3})\b", lower)
    if below:
        return None, max(0, min(100, int(below.group(1))))

    if any(token in lower for token in ["popular", "mainstream", "hits", "hit songs"]):
        return 60, None
    if any(token in lower for token in ["obscure", "underground", "less popular"]):
        return None, 40

    return None, None


 # Extracts year, year range, or decade constraints from the request.
def _extract_year_info(lower: str) -> Tuple[Optional[int], Optional[Tuple[int, int]], Optional[str]]:
    year = None
    year_range = None
    decade = None

    range_match = re.search(
        r"\b(?:from|between)\s+(19\d{2}|20\d{2})\s*(?:and|to|-)\s*(19\d{2}|20\d{2})\b",
        lower,
    )
    if range_match:
        y1 = int(range_match.group(1))
        y2 = int(range_match.group(2))
        year_range = (min(y1, y2), max(y1, y2))
        return year, year_range, decade

    decade_match = re.search(r"\b(19\d0s|20\d0s|90s|80s|70s)\b", lower)
    if decade_match:
        decade = decade_match.group(1)
        return year, year_range, decade

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", lower)
    if year_match:
        year = int(year_match.group(1))

    return year, year_range, decade

# Extracts limits on how many tracks can come from the same artist.
def _extract_max_tracks_per_artist(lower: str) -> Optional[int]:
    if "one song per artist" in lower:
        return 1

    match = re.search(
        r"\b(?:max(?:imum)?|at most)\s+(\d+)\s+(?:songs?|tracks?)\s+per artist\b",
        lower,
    )
    if match:
        return int(match.group(1))

    match = re.search(r"\b(\d+)\s+(?:songs?|tracks?)\s+per artist\b", lower)
    if match:
        return int(match.group(1))

    return None


# Extracts the requested playlist ordering style.
def _extract_ordering_style(lower: str) -> str:
    for style, phrases in ORDERING_KEYWORDS.items():
        if any(phrase in lower for phrase in phrases):
            return style
    return "smooth"


# Extracts all supported audio feature ranges from the request.
def _extract_audio_ranges(
    lower: str,
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
]:
    tempo_range = _extract_tempo_range(lower)
    energy_range = _extract_energy_range(lower)
    valence_range = _extract_valence_range(lower)
    danceability_range = _extract_danceability_range(lower)
    acousticness_range = _extract_acousticness_range(lower)
    instrumentalness_range = _extract_instrumentalness_range(lower)
    liveness_range = _extract_liveness_range(lower)
    speechiness_range = _extract_speechiness_range(lower)

    return (
        tempo_range,
        energy_range,
        valence_range,
        danceability_range,
        acousticness_range,
        instrumentalness_range,
        liveness_range,
        speechiness_range,
    )


# Extracts energy constraints from the request.
def _extract_energy_range(lower: str) -> Optional[Tuple[float, float]]:
    if "high energy" in lower or "energetic" in lower:
        return (0.75, 1.0)
    if "low energy" in lower or "calm" in lower:
        return (0.0, 0.35)
    return None


# Extracts tempo constraints from the request.
def _extract_tempo_range(lower: str) -> Optional[Tuple[float, float]]:
    bpm_match = re.search(r"\b(\d{2,3})\s*(?:-|to)\s*(\d{2,3})\s*bpm\b", lower)
    if bpm_match:
        low = float(bpm_match.group(1))
        high = float(bpm_match.group(2))
        return (min(low, high), max(low, high))

    if "fast" in lower or "high tempo" in lower:
        return (120.0, 170.0)
    if "slow" in lower or "low tempo" in lower:
        return (60.0, 100.0)
    return None


# Extracts valence constraints, such as positive or sad mood.
def _extract_valence_range(lower: str) -> Optional[Tuple[float, float]]:
    if "positive" in lower or "feel-good" in lower:
        return (0.65, 1.0)
    if "sad" in lower and "no sad" not in lower:
        return (0.0, 0.35)
    return None


# Extracts danceability constraints from the request.
def _extract_danceability_range(lower: str) -> Optional[Tuple[float, float]]:
    if "danceable" in lower or "high danceability" in lower:
        return (0.7, 1.0)
    if "low danceability" in lower:
        return (0.0, 0.4)
    return None


# Extracts acousticness constraints from the request.
def _extract_acousticness_range(lower: str) -> Optional[Tuple[float, float]]:
    if "acoustic" in lower:
        return (0.6, 1.0)
    return None


# Extracts instrumentalness constraints from the request.
def _extract_instrumentalness_range(lower: str) -> Optional[Tuple[float, float]]:
    if "instrumental" in lower:
        return (0.5, 1.0)
    return None


# Extracts liveness constraints from the request.
def _extract_liveness_range(lower: str) -> Optional[Tuple[float, float]]:
    if "live sounding" in lower or "live feel" in lower:
        return (0.5, 1.0)
    return None


# Extracts speechiness constraints from the request.
def _extract_speechiness_range(lower: str) -> Optional[Tuple[float, float]]:
    if (
        "spoken word" in lower
        or "speech heavy" in lower
        or "speech-heavy" in lower
        or "rap heavy" in lower
    ):
        return (0.4, 1.0)
    return None


# Removes duplicate strings while preserving the original order.
def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


# Checks whether the request contains any mood, activity, or vibe signal.
def _has_emotion_signal(lower: str) -> bool:
    if any(re.search(rf"\b{re.escape(token)}\b", lower) for token in known_emotion_tokens()):
        return True

    for keywords in ACTIVITY_KEYWORDS.values():
        if any(keyword in lower for keyword in keywords):
            return True

    vibe_keywords = [
        "vibe", "mood", "chill", "relaxing", "calm", "sad", "happy",
        "fun", "upbeat", "energetic", "aggressive", "focus", "focused",
        "study", "party", "sleepy", "melancholy"
    ]
    return any(re.search(rf"\b{re.escape(word)}\b", lower) for word in vibe_keywords)


# Clears emotion fields when the request is neutral and contains no mood signal.
def _strip_emotion_fields_if_neutral(request: PlaylistRequest) -> PlaylistRequest:
    # If the LLM or rule parser already found activity/emotion info, keep it.
    if request.activity or request.emotions or request.emotion_weights:
        return request

    lower = request.original_text.lower()
    if not _has_emotion_signal(lower):
        request.emotions = []
        request.emotion_weights = {}
        request.activity = None
    return request

# Ensures emotion fields are only kept when there is actual emotion context.
def _normalise_emotion_fields(request: PlaylistRequest) -> PlaylistRequest:
    has_emotion_context = bool(request.activity or request.emotions)

    if not has_emotion_context:
        request.emotion_weights = {}
        request.emotions = []
        request.activity = None

    return request