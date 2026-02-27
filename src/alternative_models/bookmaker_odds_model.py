import pandas as pd
import numpy as np
from src.alternative_models.base_alt_model import BaseAlternativeModel
from src.alternative_models.model_registry import register_model
from src.main_package.model.train import train_model


@register_model("bookmaker_odds_model")
class BookmakerOddsModel(BaseAlternativeModel):
    """
    Implement the bookmaker odds model from the main package within this repository
    """

    def __init__(self):
        """
        Initialize the bookmaker odds model
        """
        super().__init__()

    def predict(
        self,
        target_tournament: str,
        target_year: int,
        male_data: bool,
        temporal_decay: float = 0.05,
    ):
        """
        Create model predictions for the selected tournament and year.

        Args:
            target_tournament (str): The name of the tournament we are targeting
            target_year (int): The year of the tournament we want to predict
            male_data (bool): Whether or not we want to perform these predictions on male data
            temporal_decay (float): The rate of temporal decay in the model
        """
        # Determine the parameters based on the tournament
        # weight matches on the target surface 20 times more for male players and 50 times more for female players, based on our parameter fitting
        weight_factor = 20 if male_data else 50

        if target_tournament == "French Open":
            clay_weight = weight_factor
            grass_weight = 1
        elif target_tournament == "Wimbledon":
            grass_weight = weight_factor
            clay_weight = 1
        else:
            clay_weight = grass_weight = 1 / weight_factor

        # Load all data
        all_data = self._get_data(male_data)

        # Filter for target tournament matches
        target_matches = all_data[
            (all_data["Tournament"] == target_tournament)
            & (all_data["match_date"].dt.year == target_year)
        ].copy()

        if len(target_matches) == 0:
            print(f"No matches found for {target_tournament} in {target_year}")
            return

        # Get the player strength map
        _, player_strength_map = train_model(
            temporal_decay=temporal_decay,
            grass_weight=grass_weight,
            clay_weight=clay_weight,
            male_data=male_data,
            return_player_strengths=True,
            tournament=target_tournament,
            tournament_year=target_year,
        )

        # Add the strengths to the tournament matches
        target_matches["winner_strength"] = target_matches["Winner"].map(
            player_strength_map
        )
        target_matches["loser_strength"] = target_matches["Loser"].map(
            player_strength_map
        )

        # Calculate the max (worst) player and fill NA values
        max_strength = np.max(
            target_matches[~pd.isna(target_matches["loser_strength"])]["loser_strength"]
        )
        target_matches[["winner_strength", "loser_strength"]] = target_matches[
            ["winner_strength", "loser_strength"]
        ].fillna(max_strength)

        # Calculate the model probability of the winner winning
        target_matches["predicted_prob_winner"] = 1 / (
            1
            + np.exp(
                target_matches["winner_strength"] - target_matches["loser_strength"]
            )
        )

        # Save the predictions
        self._save_predictions(
            target_matches[["Winner", "Loser", "predicted_prob_winner"]].reset_index(
                drop=True
            ),
            target_tournament,
            target_year,
            male_data,
        )
