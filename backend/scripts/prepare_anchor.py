from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.real_anchor import prepare_real_anchor


def main() -> None:
    artifact = prepare_real_anchor()
    print("Anchor prepared.")
    print(f"Repo: {artifact['repo_url']} @ {artifact['repo_ref']}")
    warp = artifact["warp_detection"]
    print(f"Warp detected: {warp['found']} line={warp['line']}")


if __name__ == "__main__":
    main()
