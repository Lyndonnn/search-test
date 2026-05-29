from __future__ import annotations

from utils.io import write_json


def main() -> None:
    write_json(
        "results/baselines/sft_placeholder.json",
        {
            "stage": "sft",
            "status": "placeholder",
            "note": "SFT hooks are scaffolded; production training should attach to veRL/MMSearch-R1 data.",
        },
    )


if __name__ == "__main__":
    main()

