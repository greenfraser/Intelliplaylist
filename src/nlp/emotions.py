from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

# Emotion and audio-feature mapping utilities for IntelliPlaylist.
# This file maps mood/activity words to emotion weights, then converts those
# emotion weights into Spotify-style audio feature ranges.

Range = Tuple[float, float]


# Maps recognised mood and activity words to weighted emotion categories.
EMOTION_KEYWORDS: Dict[str, Dict[str, float]] = {
    "happy": {"joy": 1.0},
    "joyful": {"joy": 1.0},
    "cheerful": {"joy": 1.0},
    "feelgood": {"joy": 1.0},
    "fun": {"joy": 0.7, "energetic": 0.3},
    "upbeat": {"joy": 0.5, "energetic": 0.5},

    "energetic": {"energetic": 1.0},
    "hype": {"energetic": 1.0},
    "pumped": {"energetic": 1.0},
    "intense": {"energetic": 0.7, "anger": 0.3},

    "calm": {"calm": 1.0},
    "chill": {"calm": 1.0},
    "relax": {"calm": 1.0},
    "relaxing": {"calm": 1.0},
    "peaceful": {"calm": 1.0},
    "soft": {"calm": 0.8},

    "focus": {"focus": 1.0},
    "focused": {"focus": 1.0},
    "study": {"focus": 1.0},
    "revision": {"focus": 1.0},
    "concentration": {"focus": 1.0},

    "sad": {"sad": 1.0},
    "melancholy": {"sad": 1.0},
    "heartbroken": {"sad": 0.8, "romantic": 0.2},
    "lonely": {"sad": 0.8, "calm": 0.2},

    "angry": {"anger": 1.0},
    "aggressive": {"anger": 0.7, "energetic": 0.3},
    "rage": {"anger": 1.0},

    "romantic": {"romantic": 1.0},
    "love": {"romantic": 1.0},
    "date": {"romantic": 0.8, "sensual": 0.2},

    "nostalgic": {"nostalgic": 1.0},
    "throwback": {"nostalgic": 1.0},
    "memories": {"nostalgic": 1.0},

    "confident": {"confident": 1.0},
    "boss": {"confident": 1.0},
    "powerful": {"confident": 0.7, "energetic": 0.3},

    "dark": {"dark": 1.0},
    "moody": {"dark": 0.6, "sad": 0.4},
    "brooding": {"dark": 0.8, "sad": 0.2},

    "dreamy": {"dreamy": 1.0},
    "ethereal": {"dreamy": 1.0},
    "floaty": {"dreamy": 1.0},

    "hopeful": {"hopeful": 1.0},
    "uplifting": {"hopeful": 0.7, "joy": 0.3},
    "inspiring": {"hopeful": 0.7, "confident": 0.3},

    "tense": {"tense": 1.0},
    "anxious": {"tense": 1.0},
    "suspenseful": {"tense": 1.0},

    "sensual": {"sensual": 1.0},
    "sexy": {"sensual": 1.0},
    "sultry": {"sensual": 1.0},

    "gym": {"energetic": 1.0},
    "workout": {"energetic": 1.0},
    "run": {"energetic": 0.9},
    "party": {"joy": 0.4, "energetic": 0.6},
    "sleep": {"calm": 0.7, "dreamy": 0.3},
}

# Defines the target audio-feature ranges associated with each emotion category.
EMOTION_PROFILES: Dict[str, Dict[str, Range]] = {
    "joy": {
        "valence": (0.70, 1.00),
        "energy": (0.55, 0.95),
        "tempo": (105.0, 150.0),
        "danceability": (0.55, 1.00),
        "acousticness": (0.00, 0.50),
        "instrumentalness": (0.00, 0.35),
        "liveness": (0.00, 0.55),
        "speechiness": (0.00, 0.45),
    },
    "energetic": {
        "valence": (0.45, 0.90),
        "energy": (0.75, 1.00),
        "tempo": (120.0, 170.0),
        "danceability": (0.50, 1.00),
        "acousticness": (0.00, 0.40),
        "instrumentalness": (0.00, 0.30),
        "liveness": (0.00, 0.65),
        "speechiness": (0.00, 0.50),
    },
    "calm": {
        "valence": (0.35, 0.75),
        "energy": (0.15, 0.45),
        "tempo": (60.0, 110.0),
        "danceability": (0.25, 0.65),
        "acousticness": (0.35, 1.00),
        "instrumentalness": (0.00, 0.70),
        "liveness": (0.00, 0.35),
        "speechiness": (0.00, 0.20),
    },
    "focus": {
        "valence": (0.30, 0.65),
        "energy": (0.20, 0.50),
        "tempo": (65.0, 115.0),
        "danceability": (0.20, 0.55),
        "acousticness": (0.20, 0.80),
        "instrumentalness": (0.10, 0.90),
        "liveness": (0.00, 0.30),
        "speechiness": (0.00, 0.15),
    },
    "sad": {
        "valence": (0.00, 0.35),
        "energy": (0.10, 0.55),
        "tempo": (60.0, 115.0),
        "danceability": (0.10, 0.50),
        "acousticness": (0.20, 1.00),
        "instrumentalness": (0.00, 0.60),
        "liveness": (0.00, 0.30),
        "speechiness": (0.00, 0.25),
    },
    "anger": {
        "valence": (0.10, 0.55),
        "energy": (0.75, 1.00),
        "tempo": (115.0, 170.0),
        "danceability": (0.30, 0.75),
        "acousticness": (0.00, 0.25),
        "instrumentalness": (0.00, 0.35),
        "liveness": (0.00, 0.60),
        "speechiness": (0.00, 0.45),
    },

    # New useful playlist moods
    "romantic": {
        "valence": (0.45, 0.85),
        "energy": (0.25, 0.65),
        "tempo": (70.0, 125.0),
        "danceability": (0.35, 0.80),
        "acousticness": (0.20, 0.85),
        "instrumentalness": (0.00, 0.45),
        "liveness": (0.00, 0.35),
        "speechiness": (0.00, 0.30),
    },
    "nostalgic": {
        "valence": (0.30, 0.70),
        "energy": (0.25, 0.65),
        "tempo": (70.0, 130.0),
        "danceability": (0.30, 0.75),
        "acousticness": (0.20, 0.85),
        "instrumentalness": (0.00, 0.55),
        "liveness": (0.00, 0.40),
        "speechiness": (0.00, 0.30),
    },
    "confident": {
        "valence": (0.45, 0.85),
        "energy": (0.60, 1.00),
        "tempo": (95.0, 160.0),
        "danceability": (0.45, 0.95),
        "acousticness": (0.00, 0.45),
        "instrumentalness": (0.00, 0.30),
        "liveness": (0.00, 0.55),
        "speechiness": (0.00, 0.60),
    },
    "dark": {
        "valence": (0.00, 0.40),
        "energy": (0.35, 0.85),
        "tempo": (70.0, 150.0),
        "danceability": (0.20, 0.70),
        "acousticness": (0.00, 0.50),
        "instrumentalness": (0.00, 0.70),
        "liveness": (0.00, 0.45),
        "speechiness": (0.00, 0.45),
    },
    "dreamy": {
        "valence": (0.35, 0.75),
        "energy": (0.15, 0.55),
        "tempo": (60.0, 120.0),
        "danceability": (0.20, 0.65),
        "acousticness": (0.20, 0.90),
        "instrumentalness": (0.20, 0.95),
        "liveness": (0.00, 0.30),
        "speechiness": (0.00, 0.20),
    },
    "hopeful": {
        "valence": (0.55, 0.95),
        "energy": (0.35, 0.80),
        "tempo": (80.0, 140.0),
        "danceability": (0.35, 0.80),
        "acousticness": (0.10, 0.75),
        "instrumentalness": (0.00, 0.50),
        "liveness": (0.00, 0.40),
        "speechiness": (0.00, 0.30),
    },
    "tense": {
        "valence": (0.10, 0.50),
        "energy": (0.50, 0.95),
        "tempo": (100.0, 170.0),
        "danceability": (0.20, 0.65),
        "acousticness": (0.00, 0.45),
        "instrumentalness": (0.10, 0.85),
        "liveness": (0.00, 0.45),
        "speechiness": (0.00, 0.40),
    },
    "sensual": {
        "valence": (0.35, 0.75),
        "energy": (0.25, 0.65),
        "tempo": (65.0, 115.0),
        "danceability": (0.45, 0.90),
        "acousticness": (0.10, 0.65),
        "instrumentalness": (0.00, 0.45),
        "liveness": (0.00, 0.35),
        "speechiness": (0.00, 0.35),
    },
}


# Converts recognised words in the user's text into normalised emotion weights.
def text_to_emotion_weights(text: str, debug: bool = False) -> Dict[str, float]:
    tokens = _tokenize(text)
    votes: MutableMapping[str, float] = {}

    if debug:
        print("\n[EMOTION] Tokens:", tokens)

    for token in tokens:
        mapping = EMOTION_KEYWORDS.get(token)
        if not mapping:
            continue

        if debug:
            print(f"[EMOTION] token '{token}' -> {mapping}")

        for emotion, weight in mapping.items():
            votes[emotion] = votes.get(emotion, 0.0) + float(weight)

    total = sum(votes.values())
    if total <= 0:
        if debug:
            print("[EMOTION] No emotion keywords recognised.")
        return {}

    normalised = {emotion: weight / total for emotion, weight in votes.items()}

    if debug:
        print("[EMOTION] Raw emotion votes:", dict(votes))
        print("[EMOTION] Normalised emotion weights:", normalised)

    return normalised


# Blends the feature ranges for each detected emotion into one target profile.
def compute_feature_ranges(
    emotion_weights: Mapping[str, float],
    debug: bool = False,
) -> Dict[str, Range]:
    if not emotion_weights:
        return {}

    feature_names = [
        "valence",
        "energy",
        "tempo",
        "danceability",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
    ]

    result: Dict[str, Range] = {}

    if debug:
        print("\n[EMOTION] Computing feature ranges from weights:", dict(emotion_weights))

    for feature in feature_names:
        low = 0.0
        high = 0.0
        total_weight = 0.0

        for emotion, weight in emotion_weights.items():
            profile = EMOTION_PROFILES.get(emotion)
            if not profile or feature not in profile:
                continue
            lo, hi = profile[feature]
            low += lo * weight
            high += hi * weight
            total_weight += weight

        if total_weight <= 0.0:
            continue

        low /= total_weight
        high /= total_weight

        if feature == "tempo":
            blended = (max(50.0, low), min(200.0, high))
        else:
            blended = (_clamp(low, 0.0, 1.0), _clamp(high, 0.0, 1.0))

        result[feature] = blended

        if debug:
            print(f"[EMOTION]  {feature:15}: {blended}")

    if debug:
        print("[EMOTION] Final blended feature ranges:", result)

    return result


# Returns all recognised emotion keywords used by the parser.
def known_emotion_tokens() -> List[str]:
    return sorted(EMOTION_KEYWORDS.keys())


# Splits text into lowercase word tokens for keyword matching.
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())

# Restricts a numeric value to a given minimum and maximum range.
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
