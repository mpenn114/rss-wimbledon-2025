import pandas as pd
import numpy as np
from .utils import define_prize_money, ModelParameters
from src.main_package.model.train import train_model
import os
from typing import Optional


def predict_tournament_prize_money(
    male_data: bool,
    model_parameters: ModelParameters,
    tournament: str,
    tournament_year: int,
    strength_forecast_only: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Predict the prize money that each player will get in the tournament

    Args:
        male_data (bool): Whether or not we want male data
        model_parameters (ModelParameters): The parameters to use in the odds
            forecasting model
        tournament (str): The tournament to predict
        tournament_year (int): The tournament year
        strength_forecast_only (bool): Whether or not to only calculate player strengths
            if, for example, tournament draw information is not available

    Returns:
        Optional[pd.DataFrame]: The predicted prize money from each player, if required
    """
    # Get the player strength map
    _, player_strength_map = train_model(
        temporal_decay=model_parameters["temporal_decay"],
        grass_weight=model_parameters["grass_weight"],
        clay_weight=model_parameters["clay_weight"],
        male_data=male_data,
        return_player_strengths=True,
        tournament=tournament,
        tournament_year=tournament_year,
    )

    # Dump the player strength maps
    suffix = "male" if male_data else "female"
    pd.Series(player_strength_map).to_csv(
        f"results/player_strengths_{suffix}_{tournament}_{tournament_year}.csv"
    )

    # Stop the pipeline here if needed
    if strength_forecast_only:
        return

    # Load the draw. We expect the CSV to contain players in the same order as the tree
    tournament_draw = load_draw(male_data, tournament, tournament_year)
    player_strengths_pd = tournament_draw["player_name"].map(player_strength_map)

    # NB: This seems odd, but high ratings are worse throughout our model due to a
    # hugely propagated sign error...
    max_strength = np.max(player_strengths_pd[~pd.isna(player_strengths_pd)])
    player_strengths = player_strengths_pd.fillna(max_strength).to_numpy()

    # Calculate the probability of each player beating each other player
    win_probabilities = 1 / (
        1 + np.exp(player_strengths[:, np.newaxis] - player_strengths[np.newaxis, :])
    )

    # Calculate the progression probabilities
    progression = np.zeros((9, 128))
    progression[0] = 1

    for round_index in range(1, 9):
        for player in range(128):
            # Find the block that the player is playing against
            block_size = 2 ** (round_index - 1)
            block_index = player // block_size
            if block_index % 2 == 0:
                block_index += 1
            else:
                block_index -= 1

            # Calculate the probability of that player reaching the next round
            progression[round_index, player] = progression[
                round_index - 1, player
            ] * np.sum(
                win_probabilities[
                    player, block_size * block_index : block_size * (block_index + 1)
                ]
                * progression[
                    round_index - 1,
                    block_size * block_index : block_size * (block_index + 1),
                ]
            )
    # Calculate and save the round reached
    round_reached_probability = progression[:-1] - progression[1:]
    round_reached_dataframe = pd.DataFrame(
        round_reached_probability.T, columns=[f"round_{x}" for x in range(8)]
    )
    round_reached_dataframe["player_name"] = tournament_draw["player_name"]
    round_reached_dataframe.to_csv(
        f"results/predicted_round_reached_dataframe_{suffix}_{tournament}_{tournament_year}.csv",
        index=False,
    )

    # Calculate prize money

    prize_money = define_prize_money()
    tournament_draw["mean_prize_money"] = np.sum(
        round_reached_probability * prize_money[:, np.newaxis], axis=0
    )
    return tournament_draw


def load_draw(male_data: bool, tournament: str, tournament_year: int) -> pd.DataFrame:
    """
    Load a demo version of the draw for testing

    Args:
        male_data (bool): Whether or not we are using the male data
        tournament (str): The tournament we are loading
        tournament_year (int): The tournament year we are loading

    Returns:
        pd.DataFrame: The draw data
    """
    if male_data:
        file_path = (
            f"src/main_package/data/draws/{tournament}_{tournament_year}_male.csv"
        )
    else:
        file_path = (
            f"src/main_package/data/draws/{tournament}_{tournament_year}_female.csv"
        )

    if not os.path.isfile(file_path):
        raise ValueError(
            f"Tournament {tournament} could not have prize-money forecasts performed because no draw information was found at {file_path}!"  # noqa: E501
        )

    return pd.read_csv(file_path)
