from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

# Script for creating the local Spotify SQLite database from the raw CSV dataset.
# It reads the dataset in chunks, normalises column names and data types,
# then inserts the cleaned track data into the database using the SQL schema.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
DB_PATH = PROJECT_ROOT / "data" / "db" / "spotify.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

print("Running file:", __file__)
print("Using schema:", SCHEMA_PATH)
print("Schema exists:", SCHEMA_PATH.exists())
print("Using CSV:", CSV_PATH)
print("CSV exists:", CSV_PATH.exists())


CHUNK_SIZE = 50000

TRACK_COLUMNS = [
    "id",
    "name",
    "album",
    "artists",
    "explicit",
    "popularity",
    "genre",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "year",
    "release_date",
]

COLUMN_CANDIDATES = {
    "id": ["id", "track_id"],
    "name": ["name", "track_name"],
    "album": ["album", "album_name", "track_album_name"],
    "artists": ["artists", "artist_name", "track_artist"],
    "explicit": ["explicit"],
    "popularity": ["popularity", "track_popularity"],
    "genre": ["genre", "track_genre"],
    "danceability": ["danceability"],
    "energy": ["energy"],
    "key": ["key"],
    "loudness": ["loudness"],
    "mode": ["mode"],
    "speechiness": ["speechiness"],
    "acousticness": ["acousticness"],
    "instrumentalness": ["instrumentalness"],
    "liveness": ["liveness"],
    "valence": ["valence"],
    "tempo": ["tempo"],
    "duration_ms": ["duration_ms", "duration"],
    "time_signature": ["time_signature"],
    "year": ["year", "release_year"],
    "release_date": ["release_date", "album_release_date"],
}

NUMERIC_INT_COLS = {
    "explicit",
    "popularity",
    "key",
    "mode",
    "duration_ms",
    "time_signature",
    "year",
}

NUMERIC_FLOAT_COLS = {
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
}


# Returns the first matching column name found in the dataset.
def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

# Converts different explicit-value formats into 0 or 1.
def parse_explicit(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "t", "yes"} else 0

# Cleans and standardises one CSV chunk so it matches the track database schema.
def normalise_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    for target, candidates in COLUMN_CANDIDATES.items():
        source = first_existing(df, candidates)
        if source is not None:
            out[target] = df[source]
        else:
            out[target] = None

    if out["year"].isna().all() and out["release_date"].notna().any():
        extracted = out["release_date"].astype(str).str.extract(r"(\d{4})")[0]
        out["year"] = extracted

    out["explicit"] = out["explicit"].apply(parse_explicit)

    for col in NUMERIC_INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    for col in NUMERIC_FLOAT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

    for col in ["id", "name", "album", "artists", "genre", "release_date"]:
        out[col] = out[col].fillna("").astype(str)

    out = out[out["id"].str.strip() != ""]
    out = out[out["name"].str.strip() != ""]
    out = out[TRACK_COLUMNS]

    return out

# Builds the SQLite database by reading the CSV, cleaning each chunk,
# and inserting the resulting track rows into the track table.
def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema_sql)

        insert_sql = """
        INSERT OR REPLACE INTO track (
            id, name, album, artists, explicit, popularity, genre,
            danceability, energy, "key", loudness, mode, speechiness,
            acousticness, instrumentalness, liveness, valence, tempo,
            duration_ms, time_signature, year, release_date
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """

        total = 0
        for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False):
            clean = normalise_chunk(chunk)
            records = list(clean.itertuples(index=False, name=None))
            conn.executemany(insert_sql, records)
            conn.commit()
            total += len(records)
            print(f"Imported {total:,} rows...")

        print(f"\nDone. Database created at: {DB_PATH.resolve()}")

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM track")
        print("Track count:", cur.fetchone()[0])

        cur.execute("SELECT MIN(popularity), MAX(popularity) FROM track")
        print("Popularity range:", cur.fetchone())

        cur.execute("SELECT COUNT(DISTINCT genre) FROM track")
        print("Distinct genres:", cur.fetchone()[0])

    finally:
        conn.close()

# Runs the import process when this script is executed directly.
if __name__ == "__main__":
    main()