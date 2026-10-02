import type { Dispatch, SetStateAction } from "react";
import type { PlaylistRequest } from "../types/playlist";

// Modal component used to get a natural-language playlist request.
// It allows the user to type their prompt, close the modal, and start playlist generation.


type PlaylistModalProps = {
  onClose: () => void;
  onGenerate: () => Promise<void>;
  request: PlaylistRequest;
  setRequest: Dispatch<SetStateAction<PlaylistRequest>>;
  isBusy?: boolean;
};

// Renders the playlist request modal and disables generation while the app is busy.
export default function PlaylistModal({
  onClose,
  onGenerate,
  request,
  setRequest,
  isBusy = false,
}: PlaylistModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="w-full max-w-2xl rounded-2xl border border-neutral-800 bg-neutral-900 p-6 shadow-2xl">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-xl font-semibold">Ask IntelliPlaylist</h2>
          <button
            onClick={onClose}
            className="rounded-lg border border-neutral-700 px-3 py-2 text-sm text-neutral-300"
          >
            Close
          </button>
        </div>

        <div className="space-y-5">
          <div>
            <p className="mb-2 text-sm text-neutral-400">
              Describe the playlist you want in natural language.
            </p>

            <label className="mb-2 block text-sm font-medium text-neutral-300">
              Your request
            </label>

            <textarea
              value={request.original_text}
              onChange={(e) =>
                setRequest((prev) => ({
                  ...prev,
                  original_text: e.target.value,
                }))
              }
              className="h-40 w-full rounded-xl border border-neutral-700 bg-neutral-950 p-3 text-white outline-none"
              placeholder="happy upbeat gym playlist, no Drake, 15 songs"
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={onGenerate}
              disabled={isBusy || !request.original_text.trim()}
              className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-60"
            >
              {isBusy ? "Generating..." : "Generate Playlist"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}