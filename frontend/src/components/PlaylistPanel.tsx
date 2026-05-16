import { useState, type ReactNode } from "react";
import type { PlaylistTrack } from "../types/playlist";

type PlaylistPanelProps = {
  tracks: PlaylistTrack[];
  isBusy?: boolean;
  isExporting?: boolean;
  onRegenerate?: () => void;
  onExportToSpotify?: () => void;
  lockedTrackIds?: string[];
  replacingTrackIds?: string[];
  onToggleLockTrack?: (track: PlaylistTrack) => void;
  onRemoveTrack?: (track: PlaylistTrack) => void;
  onReplaceTrack?: (track: PlaylistTrack) => void;
};

function formatDuration(ms: number): string {
  if (!ms) return "—";
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatDecimal(value: number): string {
  if (Number.isNaN(value)) return "—";
  return value.toFixed(2);
}

function getYoutubeSearchUrl(track: PlaylistTrack): string {
  const query = `${track.name} by ${track.artists}`;
  return `https://www.youtube.com/results?search_query=${encodeURIComponent(query).replace(/%20/g, "+")}`;
}

function isProbablySpotifyId(id: string): boolean {
  return /^[A-Za-z0-9]{22}$/.test(id);
}


function getSpotifyAppUrl(track: PlaylistTrack): string {
  if (isProbablySpotifyId(track.id)) {
    return `spotify:track:${track.id}`;
  }

  const query = `${track.name} by ${track.artists}`;
  return `spotify:search:${encodeURIComponent(query)}`;
}

function openSpotify(track: PlaylistTrack) {
  window.location.href = getSpotifyAppUrl(track);
}


function Badge({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-medium text-neutral-300">
      {children}
    </span>
  );
}

function InfoRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-xl border border-white/5 bg-neutral-950 px-4 py-3">
      <span className="text-sm text-neutral-400">{label}</span>
      <span className="text-right text-sm font-medium text-white">{value}</span>
    </div>
  );
}

function SpotifyIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4 fill-current" aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.7 0 12 0Zm5.5 17.3c-.2.3-.6.4-.9.2-2.5-1.5-5.7-1.9-9.5-1-.4.1-.7-.1-.8-.5-.1-.4.1-.7.5-.8 4.1-.9 7.7-.5 10.5 1.2.3.2.4.6.2.9Zm1.2-2.7c-.3.4-.8.5-1.2.3-2.9-1.8-7.3-2.3-10.7-1.2-.4.1-.9-.1-1-.5-.1-.4.1-.9.5-1 3.9-1.2 8.8-.7 12.1 1.4.4.2.5.7.3 1Zm.1-2.9C15.4 9.7 9.7 9.5 6.4 10.6c-.5.2-1-.1-1.2-.6-.2-.5.1-1 .6-1.2 3.8-1.2 10.1-1 14 1.3.5.3.6.9.3 1.3-.3.5-.9.6-1.3.3Z" />
    </svg>
  );
}

function YoutubeIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4 fill-current" aria-hidden="true">
      <path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.6 12 3.6 12 3.6s-7.5 0-9.4.5A3 3 0 0 0 .5 6.2 31.2 31.2 0 0 0 0 12a31.2 31.2 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.5 9.4.5 9.4.5s7.5 0 9.4-.5a3 3 0 0 0 2.1-2.1A31.2 31.2 0 0 0 24 12a31.2 31.2 0 0 0-.5-5.8ZM9.6 15.6V8.4L15.8 12l-6.2 3.6Z" />
    </svg>
  );
}

function LockIcon({ locked }: { locked: boolean }) {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
      {locked ? (
        <path d="M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z" />
      ) : (
        <path d="M9 11V8a5 5 0 0 1 9.5-2.2M6 11h12v9H6z" />
      )}
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20 12a8 8 0 1 1-2.3-5.7" />
      <path d="M20 4v6h-6" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export default function PlaylistPanel({
  tracks,
  isBusy = false,
  isExporting = false,
  onRegenerate,
  onExportToSpotify,
  lockedTrackIds = [],
  replacingTrackIds = [],
  onToggleLockTrack,
  onRemoveTrack,
  onReplaceTrack,
}: PlaylistPanelProps) {
  const [selectedTrack, setSelectedTrack] = useState<PlaylistTrack | null>(null);

  return (
    <>
      <section className="rounded-3xl border border-white/10 bg-neutral-900/80 p-6 shadow-2xl shadow-black/20 backdrop-blur-sm">
        <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Generated Playlist</h2>
            <p className="mt-1 text-sm text-neutral-400">
              {tracks.length > 0
                ? `${tracks.length} track${tracks.length === 1 ? "" : "s"} generated`
                : "Your generated tracks will appear here"}
            </p>
          </div>

          {tracks.length > 0 ? (
            <div className="flex flex-wrap gap-3">
              {onExportToSpotify ? (
                <button
                  type="button"
                  onClick={onExportToSpotify}
                  disabled={isBusy || isExporting}
                  className="inline-flex items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-300 transition hover:scale-[1.02] hover:bg-emerald-500/20 disabled:opacity-60"
                >
                  <SpotifyIcon />
                  {isExporting ? "Exporting..." : "Export to Spotify"}
                </button>
              ) : null}

              {onRegenerate ? (
                <button
                  onClick={onRegenerate}
                  disabled={isBusy || isExporting}
                  className="rounded-xl border border-white/10 bg-white px-4 py-2 text-sm font-semibold text-black transition hover:scale-[1.02] hover:bg-neutral-200 disabled:opacity-60"
                >
                  {isBusy ? "Regenerating..." : "Regenerate"}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        {isBusy ? (
          <div className="rounded-2xl border border-white/10 bg-neutral-950 px-5 py-6 text-neutral-400">
            Generating playlist...
          </div>
        ) : tracks.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 bg-neutral-950 px-5 py-10 text-center text-neutral-400">
            No playlist generated yet.
          </div>
        ) : (
          <div className="space-y-4">
            {tracks.map((track, index) => {
              const isLocked = lockedTrackIds.includes(track.id);
              const isReplacing = replacingTrackIds.includes(track.id);

              return (
                <div
                  key={track.id}
                  className={`group rounded-2xl border px-5 py-4 transition hover:-translate-y-0.5 hover:shadow-lg ${
                    isLocked
                      ? "border-emerald-400/40 bg-gradient-to-br from-neutral-950 via-neutral-950 to-emerald-950/30 shadow-emerald-950/20"
                      : "border-white/10 bg-gradient-to-br from-neutral-950 via-neutral-950 to-emerald-950/20 hover:border-emerald-400/40 hover:shadow-emerald-950/30"
                  }`}
                >
                  <div className="flex items-center gap-4">
                    <div className="flex shrink-0 items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/15 text-sm font-semibold text-emerald-300 ring-1 ring-emerald-400/20">
                        {index + 1}
                      </div>

                      <div className="flex flex-col overflow-hidden rounded-full border border-white/10 bg-white/[0.03] p-1">
                        {onToggleLockTrack ? (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onToggleLockTrack(track);
                            }}
                            className={`flex h-8 w-8 items-center justify-center rounded-full transition ${
                              isLocked
                                ? "bg-emerald-500/20 text-emerald-200"
                                : "text-neutral-400 hover:bg-white/10 hover:text-white"
                            }`}
                            aria-label={isLocked ? `Unlock ${track.name}` : `Lock ${track.name}`}
                            title={isLocked ? "Locked for regeneration" : "Lock for regeneration"}
                          >
                            <LockIcon locked={isLocked} />
                          </button>
                        ) : null}

                        {onReplaceTrack ? (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onReplaceTrack(track);
                            }}
                            disabled={isBusy || isExporting || isReplacing}
                            className="flex h-8 w-8 items-center justify-center rounded-full text-neutral-400 transition hover:bg-sky-500/15 hover:text-sky-300 disabled:opacity-40"
                            aria-label={`Replace ${track.name}`}
                            title="Replace this song"
                          >
                            {isReplacing ? (
                              <span className="h-4 w-4 animate-spin rounded-full border-2 border-sky-300/30 border-t-sky-300" />
                            ) : (
                              <RefreshIcon />
                            )}
                          </button>
                        ) : null}

                        {onRemoveTrack ? (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onRemoveTrack(track);
                            }}
                            disabled={isBusy || isExporting}
                            className="flex h-8 w-8 items-center justify-center rounded-full text-neutral-400 transition hover:bg-red-500/15 hover:text-red-300 disabled:opacity-40"
                            aria-label={`Remove ${track.name}`}
                            title="Remove this song"
                          >
                            <XIcon />
                          </button>
                        ) : null}
                      </div>
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="truncate text-lg font-semibold text-white">
                        {track.name}
                      </div>
                      <div className="truncate text-sm text-neutral-400">
                        {track.artists}
                      </div>
                      <div className="truncate text-sm text-neutral-500">
                        {track.album || "Unknown album"}
                      </div>

                      {isLocked ? (
                        <div className="mt-2 inline-flex rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-300">
                          Locked
                        </div>
                      ) : null}
                    </div>

                    <div className="flex shrink-0 items-center gap-2">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          openSpotify(track);
                        }}
                        className="flex h-9 w-9 items-center justify-center rounded-full border border-emerald-500/20 bg-emerald-500/10 text-emerald-300 transition hover:bg-emerald-500/20"
                        aria-label={`Open ${track.name} by ${track.artists} in Spotify`}
                        title="Open in Spotify app"
                      >
                        <SpotifyIcon />
                      </button>

                      <a
                        href={getYoutubeSearchUrl(track)}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="flex h-9 w-9 items-center justify-center rounded-full border border-red-500/20 bg-red-500/10 text-red-300 transition hover:bg-red-500/20"
                        aria-label={`Search ${track.name} by ${track.artists} on YouTube`}
                        title="Search on YouTube"
                      >
                        <YoutubeIcon />
                      </a>

                      <button
                        type="button"
                        onClick={() => setSelectedTrack(track)}
                        className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/5 text-sm font-semibold text-neutral-200 transition hover:bg-white/10 hover:text-white"
                        aria-label={`View info for ${track.name}`}
                        title="View track info"
                      >
                        i
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {selectedTrack ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-2xl rounded-2xl border border-neutral-800 bg-neutral-900 p-6 shadow-2xl">
            <div className="mb-5 flex items-start justify-between gap-4">
              <div className="min-w-0">
                <h3 className="truncate text-xl font-semibold text-white">
                  {selectedTrack.name}
                </h3>
                <p className="mt-1 truncate text-sm text-neutral-400">
                  {selectedTrack.artists}
                </p>
                <p className="truncate text-sm text-neutral-500">
                  {selectedTrack.album || "Unknown album"}
                </p>
              </div>

              <button
                onClick={() => setSelectedTrack(null)}
                className="rounded-lg border border-neutral-700 px-3 py-2 text-sm text-neutral-300 transition hover:bg-neutral-800"
              >
                Close
              </button>
            </div>

            <div className="mb-5 flex flex-wrap gap-2">
              {selectedTrack.genre ? <Badge>{selectedTrack.genre}</Badge> : null}
              {selectedTrack.year ? <Badge>{selectedTrack.year}</Badge> : null}
              {selectedTrack.explicit ? <Badge>Explicit</Badge> : <Badge>Clean</Badge>}
            </div>

            <div className="space-y-3">
              <InfoRow label="Duration" value={formatDuration(selectedTrack.duration_ms)} />
              <InfoRow label="Popularity" value={selectedTrack.popularity} />
              <InfoRow label="Tempo" value={`${Math.round(selectedTrack.tempo)} BPM`} />
              <InfoRow label="Energy" value={formatDecimal(selectedTrack.energy)} />
              <InfoRow label="Valence" value={formatDecimal(selectedTrack.valence)} />
              <InfoRow
                label="Danceability"
                value={formatDecimal(selectedTrack.danceability)}
              />
              <InfoRow
                label="Acousticness"
                value={formatDecimal(selectedTrack.acousticness)}
              />
              <InfoRow
                label="Instrumentalness"
                value={formatDecimal(selectedTrack.instrumentalness)}
              />
              <InfoRow label="Liveness" value={formatDecimal(selectedTrack.liveness)} />
              <InfoRow
                label="Speechiness"
                value={formatDecimal(selectedTrack.speechiness)}
              />
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
