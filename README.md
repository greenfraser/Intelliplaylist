# IntelliPlaylist

*A Smart Assistant for Personalised Music Playlists using Answer Set Programming and Natural Language Processing*

IntelliPlaylist turns a plain-English request such as *"something upbeat for a morning run, no explicit tracks"* into a playlist that satisfies every constraint in it. A language model interprets the request, and an Answer Set Programming (ASP) solver selects the tracks, so every song in the playlist can be traced back to a rule it satisfies.

Final-year MComp Computer Science dissertation, University of Sheffield, awarded a First.


## How it works

1. **Request parsing (NLP).** The user's request is sent to the OpenAI API, which extracts structured constraints (mood/emotion, genre, energy, tempo, explicit content, length) into a fixed schema.
2. **Fact generation.** Candidate tracks from a SQLite database of Spotify tracks and their audio features are converted into ASP facts.
3. **Solving (ASP).** Clingo combines those facts with hand-written rules that map emotions to audio features and enforce the user's constraints, then returns an optimal playlist.
4. **Interface.** A Flask API serves results to a React frontend, and finished playlists can be exported straight to the user's Spotify account.

Separating interpretation (LLM) from selection (ASP) means the playlist is explainable and guaranteed to satisfy the stated constraints, rather than being whatever a language model happens to suggest.

## Tech stack

- **Reasoning:** Answer Set Programming with [Clingo](https://potassco.org/clingo/)
- **NLP:** OpenAI API
- **Backend:** Python, Flask, SQLite
- **Frontend:** React, TypeScript, Vite

## Project structure

```
backend/      Flask API
frontend/     React + TypeScript interface
src/
  asp/        Fact generation, ASP rules (rules/*.lp) and Clingo solver integration
  nlp/        Request parsing, emotion mapping and the request schema
  data/       Database access
scripts/      Data import scripts
sql/          Database schema
```

## Running locally

**1. Data.** Download the Spotify Tracks dataset from Kaggle and place it at `data/raw/dataset.csv`, then build the database:

```bash
python scripts/import_spotify.py
```

**2. Backend**

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here
# Optional, for exporting playlists to Spotify:
export SPOTIFY_CLIENT_ID=your-client-id
export SPOTIFY_CLIENT_SECRET=your-client-secret
python backend/app.py
```

The API runs on `http://127.0.0.1:5001`.

**3. Frontend**

```bash
cd frontend
npm install
npm run dev
```

Then open `http://127.0.0.1:5173`.

## Dissertation

[Read the dissertation (PDF)](docs/dissertation.pdf) -->
