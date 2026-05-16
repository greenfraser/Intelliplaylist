import { useEffect, useMemo, useState } from "react";
import Header from "../components/Header";
import PlaylistModal from "../components/PlaylistModal";
import PlaylistPanel from "../components/PlaylistPanel";
import RequestPanel from "../components/RequestPanel";
import {
  exportPlaylistToSpotify,
  generatePlaylist,
  parseRequest,
  SpotifyAuthRequiredError,
} from "../services/api";
import type { PlaylistRequest, PlaylistTrack } from "../types/playlist";

const EMPTY_REQUEST: PlaylistRequest = {
  original_text: "",
  emotions: [],
  emotion_weights: {},
  activity: null,
  constraint_explanation: null,

  playlist_size: null,
  duration_minutes: null,
  duration_range_ms: null,

  include_artists: [],
  exclude_artists: [],
  artist_target_ratio: null,
  include_albums: [],
  exclude_albums: [],
  include_tracks: [],
  exclude_tracks: [],
  include_title_terms: [],
  exclude_title_terms: [],
  include_genres: [],
  exclude_genres: [],

  min_popularity: null,
  max_popularity: null,

  year: null,
  year_range: null,
  decade: null,

  exclude_explicit: false,
  exclude_live: false,
  exclude_remix: false,
  exclude_christmas: false,

  no_repeat_artists: false,
  max_tracks_per_artist: null,

  tempo_range: null,
  energy_range: null,
  valence_range: null,
  danceability_range: null,
  acousticness_range: null,
  instrumentalness_range: null,
  liveness_range: null,
  speechiness_range: null,

  ordering_style: "smooth",
};

const PENDING_SPOTIFY_EXPORT_KEY = "intelliplaylist_pending_spotify_export";

type PendingSpotifyExport = {
  playlistName: string;
  tracks: PlaylistTrack[];
};

function hasRequestContent(request: PlaylistRequest): boolean {
  return Boolean(
    request.original_text.trim() ||
      request.constraint_explanation ||
      request.emotions.length ||
      request.activity ||
      request.playlist_size ||
      request.duration_minutes ||
      request.duration_range_ms ||
      request.include_artists.length ||
      request.exclude_artists.length ||
      request.include_albums.length ||
      request.exclude_albums.length ||
      request.include_tracks.length ||
      request.exclude_tracks.length ||
      request.include_title_terms.length ||
      request.exclude_title_terms.length ||
      request.include_genres.length ||
      request.exclude_genres.length ||
      request.min_popularity !== null ||
      request.max_popularity !== null ||
      request.year ||
      request.year_range ||
      request.decade ||
      request.no_repeat_artists ||
      request.max_tracks_per_artist !== null ||
      request.tempo_range ||
      request.energy_range ||
      request.valence_range ||
      request.danceability_range ||
      request.acousticness_range ||
      request.instrumentalness_range ||
      request.liveness_range ||
      request.speechiness_range ||
      request.exclude_explicit ||
      request.exclude_live ||
      request.exclude_remix ||
      request.exclude_christmas ||
      request.ordering_style !== "smooth",
  );
}

function buildSpotifyPlaylistName(request: PlaylistRequest): string {
  const prompt = request.original_text.trim();
  if (!prompt) return "IntelliPlaylist";

  const shortened = prompt.length > 45 ? `${prompt.slice(0, 45).trim()}…` : prompt;
  return `IntelliPlaylist - ${shortened}`;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter((value) => value.trim())));
}

export default function HomePage() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isSpotifyNameModalOpen, setIsSpotifyNameModalOpen] = useState(false);
  const [spotifyPlaylistNameDraft, setSpotifyPlaylistNameDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showConstraints, setShowConstraints] = useState(false);
  const [generationNote, setGenerationNote] = useState<string | null>(null);
  const [spotifyExportMessage, setSpotifyExportMessage] = useState<string | null>(null);
  const [lockedTrackIds, setLockedTrackIds] = useState<string[]>([]);
  const [excludedTrackNames, setExcludedTrackNames] = useState<string[]>([]);
  const [replacingTrackIds, setReplacingTrackIds] = useState<string[]>([]);

  const [parsedRequest, setParsedRequest] = useState<PlaylistRequest>(EMPTY_REQUEST);
  const [draftRequest, setDraftRequest] = useState<PlaylistRequest>(EMPTY_REQUEST);
  const [tracks, setTracks] = useState<PlaylistTrack[]>([]);

  const [isExportingSpotify, setIsExportingSpotify] = useState(false);

  const canGenerate = useMemo(() => hasRequestContent(parsedRequest), [parsedRequest]);
  const hasStarted = hasRequestContent(parsedRequest) || tracks.length > 0 || isBusy;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const spotifyStatus = params.get("spotify");

    if (!spotifyStatus) return;

    window.history.replaceState({}, "", window.location.pathname);

    if (spotifyStatus !== "connected") {
      setError("Spotify connection failed. Please try again.");
      return;
    }

    const pendingRaw = sessionStorage.getItem(PENDING_SPOTIFY_EXPORT_KEY);

    if (!pendingRaw) {
      setGenerationNote("Spotify connected. You can now export your playlist.");
      return;
    }

    sessionStorage.removeItem(PENDING_SPOTIFY_EXPORT_KEY);

    let pending: PendingSpotifyExport | null = null;

    try {
      pending = JSON.parse(pendingRaw) as PendingSpotifyExport;
    } catch {
      setError("Spotify connected, but the saved playlist could not be restored.");
      return;
    }

    if (!pending.tracks?.length) {
      setError("Spotify connected, but there was no saved playlist to export.");
      return;
    }

    setTracks(pending.tracks);
    setGenerationNote("Spotify connected. Exporting your playlist to Spotify...");
    setIsExportingSpotify(true);

    exportPlaylistToSpotify(pending.tracks, pending.playlistName || "IntelliPlaylist")
      .then((result) => {
        setGenerationNote(
          `Exported ${result.matched_count} of ${result.requested_count} tracks to Spotify.`,
        );

        if (result.playlist_url) {
          window.location.href = result.playlist_url;
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to export playlist to Spotify.");
      })
      .finally(() => {
        setIsExportingSpotify(false);
      });
  }, []);


  function handleOpenModal() {
    setDraftRequest(hasRequestContent(parsedRequest) ? { ...parsedRequest } : { ...EMPTY_REQUEST });
    setError(null);
    setIsModalOpen(true);
  }

  async function handleGenerateFromModal() {
    if (!draftRequest.original_text.trim()) {
      setError("Please enter a playlist request first.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      setSpotifyExportMessage(null);

      const parsed = await parseRequest(draftRequest.original_text);
      const result = await generatePlaylist(parsed);

      setParsedRequest(result.request);
      setDraftRequest(result.request);
      setTracks(result.tracks);
      setLockedTrackIds([]);
      setExcludedTrackNames([]);
      setGenerationNote(result.generation_note ?? null);
      setIsModalOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate playlist.");
    } finally {
      setIsBusy(false);
    }
  }

  async function handleGenerateFromHeader() {
    if (!canGenerate) {
      setError("Create or parse a playlist request first.");
      return;
    }

    try {
      setIsBusy(true);
      setError(null);
      setSpotifyExportMessage(null);

      const lockedIds = lockedTrackIds.filter((id) =>
        tracks.some((track) => track.id === id),
      );

      const lockedCount = lockedIds.length;
      const targetSize = tracks.length || parsedRequest.playlist_size || 10;
      const replacementCount = Math.max(0, targetSize - lockedCount);

      if (replacementCount === 0) {
        setGenerationNote("All tracks are locked, so nothing was regenerated.");
        return;
      }

      const replacementRequest: PlaylistRequest = {
        ...parsedRequest,

        // Only generate enough songs to fill the unlocked slots.
        playlist_size: replacementCount,
        duration_minutes: null,

        // Do NOT use include_tracks for locked songs.
        // We keep locked songs manually in the frontend.
        include_tracks: [],

        // Prevent current/removed songs from coming back as replacements.
        exclude_tracks: uniqueStrings([
          ...parsedRequest.exclude_tracks,
          ...excludedTrackNames,
          ...tracks.map((track) => track.name),
        ]),
      };

      const result = await generatePlaylist(replacementRequest);
      const replacements = [...result.tracks];

      const nextTracks = tracks.map((track) => {
        if (lockedIds.includes(track.id)) {
          return track;
        }

        return replacements.shift() ?? track;
      });

      setTracks(nextTracks);
      setGenerationNote(
        lockedCount > 0
          ? `Regenerated unlocked tracks and kept ${lockedCount} locked track${lockedCount === 1 ? "" : "s"}.`
          : result.generation_note ?? null,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate playlist.");
    } finally {
      setIsBusy(false);
    }
  }

  function handleExportToSpotify() {
    if (tracks.length === 0) {
      setError("Generate a playlist before exporting to Spotify.");
      return;
    }

    setError(null);
    setSpotifyPlaylistNameDraft(buildSpotifyPlaylistName(parsedRequest));
    setIsSpotifyNameModalOpen(true);
  }

  async function confirmExportToSpotify() {
    const playlistName = spotifyPlaylistNameDraft.trim();

    if (!playlistName) {
      setError("Please enter a playlist name.");
      return;
    }

    try {
      setIsExportingSpotify(true);
      setError(null);
      setIsSpotifyNameModalOpen(false);

      const result = await exportPlaylistToSpotify(tracks, playlistName);

      setGenerationNote(
        `Exported ${result.matched_count} of ${result.requested_count} tracks to Spotify as “${playlistName}”.`,
      );

      if (result.playlist_url) {
        window.location.href = result.playlist_url;
      }
    } catch (err) {
      if (err instanceof SpotifyAuthRequiredError) {
        sessionStorage.setItem(
          PENDING_SPOTIFY_EXPORT_KEY,
          JSON.stringify({
            playlistName,
            tracks,
          }),
        );

        window.location.href = err.authUrl;
        return;
      }

      setError(err instanceof Error ? err.message : "Failed to export playlist to Spotify.");
    } finally {
      setIsExportingSpotify(false);
    }
  }
  

  function handleToggleLockTrack(track: PlaylistTrack) {
    setLockedTrackIds((prev) =>
      prev.includes(track.id)
        ? prev.filter((id) => id !== track.id)
        : [...prev, track.id],
    );
  }

  function handleRemoveTrack(track: PlaylistTrack) {
    setTracks((prev) => prev.filter((item) => item.id !== track.id));
    setLockedTrackIds((prev) => prev.filter((id) => id !== track.id));
    setExcludedTrackNames((prev) => uniqueStrings([...prev, track.name]));
  }

  async function handleReplaceTrack(track: PlaylistTrack) {
    const trackIndex = tracks.findIndex((item) => item.id === track.id);

    if (trackIndex === -1) {
      return;
    }

    const replacementRequest: PlaylistRequest = {
      ...parsedRequest,

      // Only generate one replacement.
      playlist_size: 1,
      duration_minutes: null,

      // Do not force the rest of the playlist through include_tracks.
      include_tracks: [],

      // Do not allow the current playlist songs to come back as the replacement.
      exclude_tracks: uniqueStrings([
        ...parsedRequest.exclude_tracks,
        ...excludedTrackNames,
        ...tracks.map((item) => item.name),
      ]),
    };

    try {
      setError(null);
      setReplacingTrackIds((prev) => uniqueStrings([...prev, track.id]));

      const result = await generatePlaylist(replacementRequest);
      const replacement = result.tracks[0];

      if (!replacement) {
        throw new Error("Could not find a replacement song.");
      }

      setTracks((prev) =>
        prev.map((item, index) => (index === trackIndex ? replacement : item)),
      );

      setExcludedTrackNames((prev) => uniqueStrings([...prev, track.name]));
      setLockedTrackIds((prev) => prev.filter((id) => id !== track.id));

      setGenerationNote(`Replaced “${track.name}” with “${replacement.name}”.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to replace track.");
    } finally {
      setReplacingTrackIds((prev) => prev.filter((id) => id !== track.id));
    }
  }
    

  return (
    <div className="min-h-screen bg-neutral-950 text-white">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Header
          onNewPlaylist={handleOpenModal}
          onGeneratePlaylist={handleGenerateFromHeader}
          isBusy={isBusy}
        />

        {error ? (
          <div className="mt-6 rounded-2xl border border-red-900/60 bg-red-950/40 px-4 py-3 text-sm text-red-200 shadow-lg">
            {error}
          </div>
        ) : null}

        {spotifyExportMessage ? (
          <div className="mt-6 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100 shadow-lg">
            {spotifyExportMessage}
          </div>
        ) : null}

        {generationNote ? (
          <div className="mt-6 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100 shadow-lg">
            {generationNote}
          </div>
        ) : null}

        <div className="mt-6 space-y-6">
          {parsedRequest.original_text.trim() ? (
            <section className="rounded-3xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/10 via-neutral-900 to-neutral-950 px-6 py-5 shadow-2xl shadow-emerald-950/20">
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-emerald-500/15 text-lg">
                  ✦
                </div>

                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-400/80">
                    Playlist request
                  </p>
                  <p className="mt-2 text-lg font-semibold leading-7 text-white">
                    “{parsedRequest.original_text}”
                  </p>
                </div>
              </div>
            </section>
          ) : null}

          {!hasStarted ? (
            <section className="flex min-h-[55vh] items-center justify-center">
              <div className="w-full max-w-3xl rounded-[2rem] border border-white/10 bg-gradient-to-br from-neutral-900 via-neutral-950 to-emerald-950/20 px-8 py-14 text-center shadow-2xl shadow-black/30">
                <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-3xl bg-emerald-500/15 text-3xl ring-1 ring-emerald-400/20">
                  ✦
                </div>

                <h2 className="text-4xl font-bold tracking-tight text-white">
                  What do you want to listen to?
                </h2>

                <p className="mx-auto mt-4 max-w-xl text-base leading-7 text-neutral-400">
                  Describe a mood, activity, artist, genre, duration, or any constraints.
                  IntelliPlaylist will parse your request and generate a playlist for you.
                </p>

                <button
                  onClick={handleOpenModal}
                  disabled={isBusy}
                  className="mt-8 rounded-2xl bg-white px-8 py-4 text-lg font-bold text-black transition hover:scale-[1.03] hover:bg-neutral-200 disabled:opacity-60"
                >
                  Ask IntelliPlaylist
                </button>

                <div className="mt-8 flex flex-wrap justify-center gap-2 text-sm text-neutral-400">
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    happy gym playlist
                  </span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    chill study music
                  </span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    lots of Drake, 20 songs
                  </span>
                </div>
              </div>
            </section>
          ) : (
            <>
              <PlaylistPanel
                tracks={tracks}
                isBusy={isBusy}
                isExporting={isExportingSpotify}
                onRegenerate={handleGenerateFromHeader}
                onExportToSpotify={handleExportToSpotify}
                lockedTrackIds={lockedTrackIds}
                replacingTrackIds={replacingTrackIds}
                onToggleLockTrack={handleToggleLockTrack}
                onRemoveTrack={handleRemoveTrack}
                onReplaceTrack={handleReplaceTrack}
              />

              {hasRequestContent(parsedRequest) ? (
                <section className="rounded-3xl border border-white/10 bg-neutral-900/60 p-4 shadow-2xl shadow-black/20">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <h2 className="text-lg font-semibold text-white">Request interpretation</h2>
                      <p className="mt-1 text-sm text-neutral-400">
                        View the parsed constraints and explanation for this playlist request.
                      </p>
                    </div>

                    <button
                      onClick={() => setShowConstraints((value) => !value)}
                      className="rounded-xl border border-white/10 bg-neutral-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-900"
                    >
                      {showConstraints ? "Hide parsed constraints" : "View parsed constraints"}
                    </button>
                  </div>

                  {showConstraints ? (
                    <div className="mt-4">
                      <RequestPanel request={parsedRequest} />
                    </div>
                  ) : null}
                </section>
              ) : null}
            </>
          )}
        </div>
      </div>

      {isSpotifyNameModalOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-md rounded-3xl border border-white/10 bg-neutral-950 p-6 shadow-2xl">
            <div className="mb-5">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-400/80">
                Spotify export
              </p>
              <h2 className="mt-2 text-2xl font-bold text-white">
                Name your playlist
              </h2>
              <p className="mt-2 text-sm text-neutral-400">
                Choose the name that will appear in Spotify.
              </p>
            </div>

            <label className="block text-sm font-medium text-neutral-300">
              Playlist name
            </label>
            <input
              type="text"
              value={spotifyPlaylistNameDraft}
              onChange={(e) => setSpotifyPlaylistNameDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  void confirmExportToSpotify();
                }
              }}
              placeholder="My IntelliPlaylist"
              autoFocus
              className="mt-2 w-full rounded-2xl border border-white/10 bg-neutral-900 px-4 py-3 text-white outline-none transition focus:border-emerald-400/40 focus:ring-2 focus:ring-emerald-500/20"
            />

            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setIsSpotifyNameModalOpen(false)}
                className="rounded-2xl border border-white/10 bg-neutral-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-neutral-800"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={() => void confirmExportToSpotify()}
                className="rounded-2xl bg-emerald-500 px-5 py-3 text-sm font-bold text-black transition hover:scale-[1.02] hover:bg-emerald-400"
              >
                Export to Spotify
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {isModalOpen ? (
        <PlaylistModal
          onClose={() => setIsModalOpen(false)}
          onGenerate={handleGenerateFromModal}
          request={draftRequest}
          setRequest={setDraftRequest}
          isBusy={isBusy}
        />
      ) : null}
    </div>
  );
}
