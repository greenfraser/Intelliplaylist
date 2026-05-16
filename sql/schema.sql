CREATE TABLE track (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    album             TEXT,
    artists           TEXT,
    explicit          BOOLEAN,
    popularity        INTEGER,
    genre             TEXT,
    danceability      REAL,
    energy            REAL,
    "key"             INTEGER,
    loudness          REAL,
    mode              INTEGER,
    speechiness       REAL,
    acousticness      REAL,
    instrumentalness  REAL,
    liveness          REAL,
    valence           REAL,
    tempo             REAL,
    duration_ms       INTEGER,
    time_signature    INTEGER,
    year              INTEGER,
    release_date      TEXT
);

CREATE INDEX idx_track_name ON track(name);
CREATE INDEX idx_track_album ON track(album);
CREATE INDEX idx_track_artists ON track(artists);
CREATE INDEX idx_track_genre ON track(genre);
CREATE INDEX idx_track_popularity ON track(popularity);
CREATE INDEX idx_track_year ON track(year);