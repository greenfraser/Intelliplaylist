import type { PlaylistRequest, PlaylistTrack } from "../types/playlist";

// API helper functions for communicating with the Flask backend.
// This file handles request parsing, playlist generation, Spotify export,
// and backend error handling for the frontend.

const API_BASE = "http://127.0.0.1:5001/api";

// Custom error used when Spotify export requires the user to log in first.
export class SpotifyAuthRequiredError extends Error {
  authUrl: string;

  constructor(authUrl: string) {
    super("Spotify login required.");
    this.name = "SpotifyAuthRequiredError";
    this.authUrl = authUrl;
  }
}

// Shape of the response returned after exporting a playlist to Spotify.
export type SpotifyExportResult = {
  playlist_id: string;
  playlist_url: string;
  requested_count: number;
  matched_count: number;
  missing_tracks: string[];
  matched_tracks: Array<{
    original_name: string;
    original_artists: string;
    spotify_name: string;
    spotify_artists: string;
    spotify_uri: string;
    spotify_url: string;
  }>;
};

// Reads a JSON response from the backend and throws an error if the request failed.
async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json();
  if (!response.ok) {
    const message = typeof payload?.error === "string" ? payload.error : "Request failed.";
    throw new Error(message);
  }
  return payload as T;
}

// Sends the user's natural-language request to the backend parser.
export async function parseRequest(text: string): Promise<PlaylistRequest> {
  const response = await fetch(`${API_BASE}/parse-request`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ text }),
  });

  const payload = await readJson<{ request: PlaylistRequest }>(response);
  return payload.request;
}

// Sends a parsed playlist request to the backend and returns the generated tracks.
export async function generatePlaylist(
  request: PlaylistRequest,
): Promise<{ request: PlaylistRequest; tracks: PlaylistTrack[]; generation_note: string | null }> {
  const response = await fetch(`${API_BASE}/generate-playlist`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ request }),
  });

  return readJson<{ request: PlaylistRequest; tracks: PlaylistTrack[]; generation_note: string | null }>(response);
}

// Exports the generated playlist to Spotify, or redirects the user to Spotify login if needed.
export async function exportPlaylistToSpotify(
  tracks: PlaylistTrack[],
  playlistName = "IntelliPlaylist",
): Promise<SpotifyExportResult> {
  const response = await fetch(`${API_BASE}/export-spotify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({
      playlist_name: playlistName,
      tracks,
    }),
  });

  const payload = await response.json();

  if (response.status === 401 && typeof payload?.auth_url === "string") {
    throw new SpotifyAuthRequiredError(payload.auth_url);
  }

  if (!response.ok) {
    const message = typeof payload?.error === "string" ? payload.error : "Failed to export playlist.";
    throw new Error(message);
  }

  return payload as SpotifyExportResult;
}
