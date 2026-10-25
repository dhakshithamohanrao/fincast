"""Full rebuild: ingestion first, then training."""
from pipelines.run_training import run_module


def main() -> None:
    for module in ("pipelines.run_ingestion", "pipelines.run_training"):
        print(f"\n##### {module} #####", flush=True)
        if run_module(module) != 0:
            raise SystemExit(f"Failed: {module}")
    print("\nFull rebuild finished.")


if __name__ == "__main__":
    main()