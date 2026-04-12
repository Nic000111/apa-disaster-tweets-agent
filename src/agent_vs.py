import argparse


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

    raise AssertionError(f"Unhandled agent choice: {args.agent}")


if __name__ == "__main__":
    raise SystemExit(main())

