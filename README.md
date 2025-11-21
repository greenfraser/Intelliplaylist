# Dissertation – Spotify Playlist Generator with ASP

## Overview

- Uses a large Spotify tracks dataset.
- Stores data in a SQLite database.
- Uses Answer Set Programming (ASP) to generate playlists based on user requests.
- Simple NLP layer maps natural language requests to ASP constraints.

## Project layout

- `data/` – raw, processed data and the SQLite database.
- `sql/` – SQL schema(s).
- `src/` – Python package with:
  - `data/` – database access, loading.
  - `asp/` – fact generation, ASP rules, solver integration.
  - `nlp/` – intent parsing and mapping to constraints.
  - `ui/` – endpoints or logic for displaying playlists.
- `scripts/` – one-off scripts (importing data, etc.).
- `notebooks/` – experiments, EDA.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

