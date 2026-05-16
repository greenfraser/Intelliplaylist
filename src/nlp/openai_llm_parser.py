from pydantic import BaseModel, Field
from openai import OpenAI
import json

client = OpenAI()


class EmotionWeights(BaseModel):
    joy: float = 0.0
    energetic: float = 0.0
    calm: float = 0.0
    focus: float = 0.0
    sad: float = 0.0
    anger: float = 0.0
    romantic: float = 0.0
    nostalgic: float = 0.0
    confident: float = 0.0
    dark: float = 0.0
    dreamy: float = 0.0
    hopeful: float = 0.0
    tense: float = 0.0
    sensual: float = 0.0


class ParsedPlaylistRequest(BaseModel):
    original_text: str
    emotions: list[str] = Field(default_factory=list)
    emotion_weights: EmotionWeights = Field(default_factory=EmotionWeights)
    activity: str | None = None
    constraint_explanation: str | None = None

    include_artists: list[str] = Field(default_factory=list)
    exclude_artists: list[str] = Field(default_factory=list)
    artist_target_ratio: float | None = None
    include_albums: list[str] = Field(default_factory=list)
    exclude_albums: list[str] = Field(default_factory=list)
    include_tracks: list[str] = Field(default_factory=list)
    exclude_tracks: list[str] = Field(default_factory=list)
    include_title_terms: list[str] = Field(default_factory=list)
    exclude_title_terms: list[str] = Field(default_factory=list)
    include_genres: list[str] = Field(default_factory=list)
    exclude_genres: list[str] = Field(default_factory=list)

    playlist_size: int | None = None
    duration_minutes: int | None = None
    duration_range_ms: list[int] | None = None
    min_popularity: int | None = None
    max_popularity: int | None = None
    year: int | None = None
    year_range: list[int] | None = None
    decade: str | None = None

    exclude_explicit: bool = False
    exclude_live: bool = False
    exclude_remix: bool = False
    exclude_christmas: bool = False
    no_repeat_artists: bool = False
    max_tracks_per_artist: int | None = None

    tempo_range: list[float] | None = None
    energy_range: list[float] | None = None
    valence_range: list[float] | None = None
    danceability_range: list[float] | None = None
    acousticness_range: list[float] | None = None
    instrumentalness_range: list[float] | None = None
    liveness_range: list[float] | None = None
    speechiness_range: list[float] | None = None

    ordering_style: str = "smooth"


def openai_llm_callable(system_prompt: str, user_text: str) -> str:
    print("\n========== OPENAI REQUEST ==========")
    print("[SYSTEM PROMPT]")
    print(system_prompt)
    print("\n[USER TEXT]")
    print(user_text)

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        text_format=ParsedPlaylistRequest,
    )

    parsed = response.output_parsed
    payload = parsed.model_dump()
    payload["emotion_weights"] = {
        key: float(value)
        for key, value in payload["emotion_weights"].items()
        if float(value) > 0.0
    }

    for name in ("min_popularity", "max_popularity"):
        if payload.get(name) is not None:
            payload[name] = max(0, min(100, int(payload[name])))

    if payload.get("constraint_explanation") is not None:
        payload["constraint_explanation"] = str(payload["constraint_explanation"]).strip() or None

    print("\n[PARSED RESPONSE]")
    print(json.dumps(payload, indent=2))

    return json.dumps(payload)
