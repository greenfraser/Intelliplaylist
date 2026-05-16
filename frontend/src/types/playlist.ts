export type EmotionWeights = Record<string, number>;

export type PlaylistRequest = {
  original_text: string;
  emotions: string[];
  emotion_weights: EmotionWeights;
  activity: string | null;
  constraint_explanation: string | null;

  playlist_size: number | null;
  duration_minutes: number | null;
  duration_range_ms: [number, number] | null;

  include_artists: string[];
  exclude_artists: string[];
  artist_target_ratio: number | null;
  include_albums: string[];
  exclude_albums: string[];
  include_tracks: string[];
  exclude_tracks: string[];
  include_title_terms: string[];
  exclude_title_terms: string[];
  include_genres: string[];
  exclude_genres: string[];

  min_popularity: number | null;
  max_popularity: number | null;

  year: number | null;
  year_range: [number, number] | null;
  decade: string | null;

  exclude_explicit: boolean;
  exclude_live: boolean;
  exclude_remix: boolean;
  exclude_christmas: boolean;

  no_repeat_artists: boolean;
  max_tracks_per_artist: number | null;

  tempo_range: [number, number] | null;
  energy_range: [number, number] | null;
  valence_range: [number, number] | null;
  danceability_range: [number, number] | null;
  acousticness_range: [number, number] | null;
  instrumentalness_range: [number, number] | null;
  liveness_range: [number, number] | null;
  speechiness_range: [number, number] | null;

  ordering_style: string;
};

export type PlaylistTrack = {
  id: string;
  name: string;
  artists: string;
  tempo: number;
  energy: number;
  valence: number;
  danceability: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  speechiness: number;
  album: string;
  explicit: boolean;
  duration_ms: number;
  year: number;
  popularity: number;
  genre: string;
};
