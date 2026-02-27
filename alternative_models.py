import argparse
from src.alternative_models.model_registry import run_model


def main():
    parser = argparse.ArgumentParser(description="Run Alternative Tournament Models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to run (e.g., baseline)",
    )
    parser.add_argument(
        "--tournament",
        type=str,
        required=True,
        help="Tournament name (e.g., Wimbledon)",
    )
    parser.add_argument("--year", type=int, required=True, help="Target year")
    parser.add_argument(
        "--female",
        action="store_false",
        dest="male",
        help="Set this flag to use female data (defaults to male)",
    )

    parser.set_defaults(male=True)
    args = parser.parse_args()

    try:
        run_model(args.model, args.tournament, args.year, args.male)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
