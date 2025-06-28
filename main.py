import argparse
from src.main_package.model.fit_parameters import fit_parameters
from src.main_package.model.predict import predict_wimbledon_prize_money
from src.main_package.model.assign_tokens import assign_tokens
from src.main_package.model.utils import ModelParameters
import pandas as pd


def run_token_pipeline(train: bool = False):
    """
    Run the full pipeline for assigning tokens

    Args:
        train (bool): Whether to fit the parameters, or run the
            main pipeline
    """
    if train:
        fit_parameters(male_data=True)
        fit_parameters(male_data=False)
        return

    # Define the parameters
    parameters = ModelParameters(temporal_decay=0.05, grass_weight=20.0)

    # Calculate the estimated prize money
    male_prize_money = predict_wimbledon_prize_money(True, parameters)

    # Define the parameters
    parameters = ModelParameters(temporal_decay=0.05, grass_weight=50.0)
    female_prize_money = predict_wimbledon_prize_money(False, parameters)

    # Concatenate the prize money from the two tournaments
    combined_prize_money = pd.concat(
        [male_prize_money, female_prize_money]
    ).sort_values(by="mean_prize_money", ascending=False)
    combined_prize_money.to_csv("combined_prize_money.csv", index=False)

    # Create the overall token assignment
    assign_tokens(combined_prize_money)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the token assignment pipeline")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training and run the token assignment pipeline instead",
    )
    args = parser.parse_args()

    run_token_pipeline(train=not args.no_train)
