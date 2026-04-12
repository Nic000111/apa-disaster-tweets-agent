import argparse
import subprocess
import sys

from config import AGENT_V2_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="agent_vs",
        description="Run a specific Disaster Tweets agent version.",
    )
    parser.add_argument(
        "agent",
        help="Which agent to run (aliases: simple=v1, transformer=advanced=v2).",
        choices=[
            "v1_simple",
            "simple",
            "v1",
            "v2_transformer",
            "transformer",
            "advanced",
            "v2",
            "v3_autonomous",
            "autonomous",
            "v3",
        ],
    )

    args = parser.parse_args()

    if args.agent in {"v1_simple", "simple", "v1"}:
        from agents import v1_simple

        v1_simple.main()
        return 0

    if args.agent in {"v2_transformer", "transformer", "advanced", "v2"}:
        from agents import v2_transformer

        v2_transformer.main()
        return 0

    if args.agent in {"v3_autonomous", "autonomous", "v3"}:
        script = AGENT_V2_DIR / "agent_fully_autonomous.py"
        if not script.exists():
            raise FileNotFoundError(f"Missing autonomous agent script: {script}")

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(AGENT_V2_DIR),
            check=False,
        )
        return completed.returncode

    raise AssertionError(f"Unhandled agent choice: {args.agent}")


if __name__ == "__main__":
    raise SystemExit(main())

