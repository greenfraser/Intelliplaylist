from __future__ import annotations

import json
from typing import List, Tuple

from src.asp.playlist import build_playlist_from_request
from src.nlp.request_parser import parse_playlist_request


def print_playlist(rows: List[Tuple], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for row in rows:
        _track_id, name, artists, tempo, energy, valence, *_rest = row
        print(
            f"- {name} — {artists} "
            f"(tempo={tempo:.1f}, energy={energy:.2f}, valence={valence:.2f})"
        )


def main() -> None:
    description = input(
        "Describe the vibe for your playlist "
        "(e.g. 'happy chill friday night'): "
    ).strip()

    if not description:
        print("Please describe the vibe, e.g. 'happy chill friday night'.")
        return

    request = parse_playlist_request(description, debug=True)

    print("\nParsed request")
    print("==============")
    print(json.dumps(request.to_dict(), indent=2))

    rows = build_playlist_from_request(request, debug=True)

    if not rows:
        print("No tracks found that match that request yet.")
        return

    print_playlist(rows, f"Playlist for: {description}")


if __name__ == "__main__":
    main()
