// Header component for the main page.
// Displays the app title, a short description, and the button used to open
// the playlist request modal.

type HeaderProps = {
  onNewPlaylist: () => void;
  onGeneratePlaylist: () => void;
  isBusy?: boolean;
};

// Renders the top page header and disables the main action button while the app is busy.
export default function Header({
  onNewPlaylist,
  isBusy = false,
}: HeaderProps) {
  return (
    <header className="rounded-3xl border border-white/10 bg-gradient-to-r from-neutral-900 to-neutral-950 px-6 py-5 shadow-2xl shadow-black/20">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">IntelliPlaylist</h1>
          <p className="mt-1 text-sm text-neutral-400">
            Build playlists from natural-language requests
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={onNewPlaylist}
            disabled={isBusy}
            className="rounded-xl border border-white/10 bg-white px-4 py-2.5 text-sm font-semibold text-black transition hover:scale-[1.02] hover:bg-neutral-200 disabled:opacity-60"
          >
            Ask IntelliPlaylist
          </button>
        </div>
      </div>
    </header>
  );
}