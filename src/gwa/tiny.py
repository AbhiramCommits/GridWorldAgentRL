"""Build the local randomly-initialized tiny GPT-2 used by the CI smoke pipeline."""

from __future__ import annotations

from pathlib import Path

from .agent.policies import _build_local_tiny_model
from .utils import REPO_ROOT


def main() -> Path:
    path = REPO_ROOT / "assets" / "tiny_model"
    _build_local_tiny_model(path)
    print(f"wrote local tiny model to {path}")
    print(f"wrote local tiny tokenizer to {path.parent / 'tiny_tokenizer'}")
    return path


if __name__ == "__main__":
    main()
