import argparse
from src.main_package.model.fit_parameters import fit_parameters
from src.main_package.model.predict import predict_tournament_prize_money
from src.main_package.model.assign_tokens import assign_tokens
from src.main_package.model.utils import ModelParameters
import pandas as pd


def run_token_pipeline(tournament:str = "Wimbledon", tournament_year:int = 2025,
                    grass_weight_male:float = 1.0,
                    grass_weight_female:float = 1.0,
                    clay_weight_male:float = 1.0,
                    clay_weight_female:float = 1.0,   
                    train: bool = False,
                       calculate_strengths_only:bool = True):
    """
    Run the full pipeline for assigning tokens

    Note that all results are assigned to results/...

    Args:
        tournament (str): The tournament to predict
        tournament_year (int): The year of the tournament
        train (bool): Whether to fit the parameters, or run the
            main pipeline
        grass_weight_male (float): The weight to put on grass matches for male players
        grass_weight_female (float): The weight to put on grass matches for female playres
        clay_weight_male (float): The weight to put on clay matches for male players
        clay_weight_female (float): The weight to put on clay matches for female players
        calculate_strengths_only (bool): Whether to run the full pipeline, or
            only calculate the player strengths
    """
    if train:
        fit_parameters(male_data=True)
        fit_parameters(male_data=False)
        return

    # Define the parameters
    parameters = ModelParameters(temporal_decay=0.05,clay_weight=clay_weight_male, grass_weight=grass_weight_male)

    # Calculate the estimated prize money
    male_prize_money = predict_tournament_prize_money(True, parameters, tournament, tournament_year,strength_forecast_only=calculate_strengths_only)

    # Define the parameters
    parameters = ModelParameters(temporal_decay=0.05,clay_weight=clay_weight_female, grass_weight=grass_weight_female)
    female_prize_money = predict_tournament_prize_money(False, parameters, tournament, tournament_year, strength_forecast_only=calculate_strengths_only)

    # Return if only calculating player strengths
    if calculate_strengths_only:
        return

    # Concatenate the prize money from the two tournaments
    combined_prize_money = pd.concat(
        [male_prize_money, female_prize_money]
    ).sort_values(by="mean_prize_money", ascending=False)
    combined_prize_money.to_csv(f"results/combined_prize_money_{tournament}_{tournament_year}.csv", index=False)

    # Create the overall token assignment
    assign_tokens(combined_prize_money)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the token assignment pipeline")
    
    parser.add_argument(
        "--tournament",
        type=str,
        default="Wimbledon",
        help="The tournament to predict (default: Wimbledon)"
    )
    
    parser.add_argument(
        "--tournament-year",
        type=int,
        default=2025,
        help="The year of the tournament (default: 2025)"
    )

    parser.add_argument(
        "--grass-weight-male",
        type=float,
        default=1.0,
        help="The weight to put on grass results for male players, relative to a weight of 1 for hard court"
    )

    parser.add_argument(
        "--grass-weight-female",
        type=float,
        default=1.0,
        help="The weight to put on grass results for female players, relative to a weight of 1 for hard court"
    )

    parser.add_argument(
        "--clay-weight-male",
        type=float,
        default=1.0,
        help="The weight to put on clay results for male players, relative to a weight of 1 for hard court"
    )

    parser.add_argument(
        "--clay-weight-female",
        type=float,
        default=1.0,
        help="The weight to put on clay results for female players, relative to a weight of 1 for hard court"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fit the parameters instead of just running the main pipeline"
    )
    
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run the full pipeline instead of just calculating player strengths"
    )

    args = parser.parse_args()

    run_token_pipeline(
        tournament=args.tournament,
        tournament_year=args.tournament_year,
        grass_weight_male=args.grass_weight_male,
        grass_weight_female=args.grass_weight_female,
        clay_weight_male=args.clay_weight_male,
        clay_weight_female=args.clay_weight_female,
        train=args.train,
        calculate_strengths_only=not args.full_pipeline
    )