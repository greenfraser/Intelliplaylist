import type { ReactNode } from "react";
import type { PlaylistRequest } from "../types/playlist";

type RequestPanelProps = {
  request: PlaylistRequest | null;
};

function Tag({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-white/10 bg-neutral-950 px-3 py-1 text-xs font-medium text-neutral-200">
      {label}
    </span>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-neutral-500">
        {title}
      </p>
      {children}
    </div>
  );
}

function TagList({ values }: { values: string[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {values.map((value) => (
        <Tag key={value} label={value} />
      ))}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-xl border border-white/5 bg-neutral-950 px-3 py-3">
      <span className="text-sm text-neutral-400">{label}</span>
      <span className="text-right text-sm font-medium text-white">{value}</span>
    </div>
  );
}

function formatPercentRange(range: [number, number]): string {
  return `${Math.round(range[0] * 100)}%–${Math.round(range[1] * 100)}%`;
}

function formatTempoRange(range: [number, number]): string {
  return `${Math.round(range[0])}–${Math.round(range[1])} BPM`;
}

function formatDurationRange(range: [number, number]): string {
  const formatMs = (ms: number) => {
    const totalSeconds = Math.round(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;

    if (seconds === 0) {
      return `${minutes} min`;
    }

    return `${minutes}:${String(seconds).padStart(2, "0")}`;
  };

  return `${formatMs(range[0])}–${formatMs(range[1])}`;
}

function WeightBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="text-neutral-200">{label}</span>
        <span className="text-neutral-400">{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-neutral-800">
        <div className="h-full rounded-full bg-emerald-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function RequestPanel({ request }: RequestPanelProps) {
  if (!request) {
    return (
      <aside className="rounded-3xl border border-white/10 bg-neutral-900/80 p-6 shadow-2xl shadow-black/20">
        <h2 className="text-2xl font-bold tracking-tight">Parsed request</h2>
        <p className="mt-4 text-neutral-400">No parsed request yet.</p>
      </aside>
    );
  }

  const emotionWeights = request.emotion_weights ?? {};
  const weightEntries = Object.entries(emotionWeights)
    .filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]);
  const hasAnythingToShow =
    Boolean(request.constraint_explanation) ||
    weightEntries.length > 0 ||
    request.emotions.length > 0 ||
    request.activity ||
    request.playlist_size !== null ||
    request.duration_minutes !== null ||
    request.duration_range_ms !== null ||
    request.include_artists.length > 0 ||
    request.exclude_artists.length > 0 ||
    request.include_albums.length > 0 ||
    request.exclude_albums.length > 0 ||
    request.include_tracks.length > 0 ||
    request.exclude_tracks.length > 0 ||
    request.include_genres.length > 0 ||
    request.exclude_genres.length > 0 ||
    request.min_popularity !== null ||
    request.max_popularity !== null ||
    request.year !== null ||
    request.year_range !== null ||
    request.decade ||
    request.no_repeat_artists ||
    request.max_tracks_per_artist !== null ||
    request.tempo_range !== null ||
    request.energy_range !== null ||
    request.valence_range !== null ||
    request.danceability_range !== null ||
    request.acousticness_range !== null ||
    request.instrumentalness_range !== null ||
    request.liveness_range !== null ||
    request.speechiness_range !== null ||
    request.exclude_explicit ||
    request.exclude_live ||
    request.exclude_remix ||
    request.exclude_christmas ||
    request.ordering_style !== "smooth";

  return (
    <aside className="rounded-3xl border border-white/10 bg-neutral-900/80 p-6 shadow-2xl shadow-black/20">
      <h2 className="text-2xl font-bold tracking-tight">Parsed request</h2>
      <p className="mt-1 text-sm text-neutral-400">View how IntelliPlaylist interpreted your prompt</p>

      {!hasAnythingToShow ? (
        <p className="mt-6 text-neutral-400">No parsed constraints yet.</p>
      ) : (
        <div className="mt-6 space-y-5">
          {request.constraint_explanation ? (
            <Section title="Why IntelliPlaylist interpreted it this way">
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-4 text-sm leading-6 text-neutral-200">
                {request.constraint_explanation}
              </div>
            </Section>
          ) : null}

          {weightEntries.length > 0 && (request.activity || request.emotions.length > 0) && (
            <Section title="Emotion weights">
              <div className="space-y-3">
                {weightEntries.map(([label, value]) => (
                  <WeightBar key={label} label={label} value={value} />
                ))}
              </div>
            </Section>
          )}

          {request.emotions.length > 0 && (
            <Section title="Raw emotion terms">
              <TagList values={request.emotions} />
            </Section>
          )}

          {request.include_artists.length > 0 && (
            <Section title="Include artists">
              <TagList values={request.include_artists} />
            </Section>
          )}

          {request.exclude_artists.length > 0 && (
            <Section title="Exclude artists">
              <TagList values={request.exclude_artists} />
            </Section>
          )}

          {request.include_albums.length > 0 && (
            <Section title="Include albums">
              <TagList values={request.include_albums} />
            </Section>
          )}

          {request.include_tracks.length > 0 && (
            <Section title="Include tracks">
              <TagList values={request.include_tracks} />
            </Section>
          )}

          {request.include_title_terms.length > 0 && (
            <Section title="Title terms">
              <TagList values={request.include_title_terms} />
            </Section>
          )}

          {request.exclude_title_terms.length > 0 && (
            <Section title="Exclude title terms">
              <TagList values={request.exclude_title_terms} />
            </Section>
          )}

          {request.include_genres.length > 0 && (
            <Section title="Include genres">
              <TagList values={request.include_genres} />
            </Section>
          )}

          {request.exclude_genres.length > 0 && (
            <Section title="Exclude genres">
              <TagList values={request.exclude_genres} />
            </Section>
          )}

          <div className="space-y-3">
            {request.activity && <InfoRow label="Activity" value={request.activity} />}
            {request.playlist_size !== null && <InfoRow label="Playlist size" value={request.playlist_size} />}
            {request.duration_minutes !== null && <InfoRow label="Duration" value={`${request.duration_minutes} minutes`} />}
            {request.min_popularity !== null && request.min_popularity !== 70 && (
              <InfoRow label="Min popularity" value={request.min_popularity} />
            )}
            {request.max_popularity !== null && <InfoRow label="Max popularity" value={request.max_popularity} />}
            {request.year !== null && <InfoRow label="Year" value={request.year} />}
            {request.year_range !== null && <InfoRow label="Year range" value={`${request.year_range[0]}–${request.year_range[1]}`} />}
            {request.decade && <InfoRow label="Decade" value={request.decade} />}
            {request.no_repeat_artists && <InfoRow label="No repeat artists" value="Yes" />}
            {request.max_tracks_per_artist !== null && <InfoRow label="Max tracks per artist" value={request.max_tracks_per_artist} />}
            {request.ordering_style !== "smooth" && <InfoRow label="Ordering" value={request.ordering_style} />}
            {request.duration_range_ms !== null && (
              <InfoRow label="Track duration" value={formatDurationRange(request.duration_range_ms)} />
            )}

            {request.tempo_range !== null && (
              <InfoRow label="Tempo range" value={formatTempoRange(request.tempo_range)} />
            )}

            {request.energy_range !== null && (
              <InfoRow label="Energy range" value={formatPercentRange(request.energy_range)} />
            )}

            {request.valence_range !== null && (
              <InfoRow label="Valence range" value={formatPercentRange(request.valence_range)} />
            )}

            {request.danceability_range !== null && (
              <InfoRow label="Danceability range" value={formatPercentRange(request.danceability_range)} />
            )}

            {request.acousticness_range !== null && (
              <InfoRow label="Acousticness range" value={formatPercentRange(request.acousticness_range)} />
            )}

            {request.instrumentalness_range !== null && (
              <InfoRow label="Instrumentalness range" value={formatPercentRange(request.instrumentalness_range)} />
            )}

            {request.liveness_range !== null && (
              <InfoRow label="Liveness range" value={formatPercentRange(request.liveness_range)} />
            )}

            {request.speechiness_range !== null && (
              <InfoRow label="Speechiness range" value={formatPercentRange(request.speechiness_range)} />
            )}
          </div>

          {(request.exclude_explicit || request.exclude_live || request.exclude_remix || request.exclude_christmas) && (
            <Section title="Exclusions">
              <div className="flex flex-wrap gap-2">
                {request.exclude_explicit && <Tag label="No explicit" />}
                {request.exclude_live && <Tag label="No live" />}
                {request.exclude_remix && <Tag label="No remixes" />}
                {request.exclude_christmas && <Tag label="No Christmas" />}
              </div>
            </Section>
          )}
        </div>
      )}
    </aside>
  );
}
