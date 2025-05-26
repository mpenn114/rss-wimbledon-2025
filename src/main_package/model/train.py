import pandas as pd
from src.main_package.data.load_data import load_data
from datetime import date
import numpy as np
from scipy.optimize import minimize
from typing import Tuple


def train_model(
    temporal_decay: float = 0.4,
    clay_weight: float = 0.8,
    hard_weight: float = 0.9,
    male_data: bool = True,
) -> float:
    """
    Train the model.

    Idea is that we assume that players can be represented by a single ability R_p,
        where we fit an additive graph such that if p_{AB} = P(A beats B)
        from the bookies data then

        log((1-p_{ab})/p_{ab}) + log((1-p_{bc})/p_{bc}) = log((1-p_{ac})/p_{ac})

        which assumes that these probabilities come from an ELO-style model.

    Args:
        temporal_decay (float): The amount of decay that the weight we place
            on inter-player matches experiences in a year
        clay_weight (float): The weight that we give to clay-court matches, relative
            to grass-court
        hard_weight (float): The weight that we give to hard-court matches, relative
            to grass-court
        male_data (bool): Whether or not we want the male data

    Returns:
        float: The average RMSE over the years 2022-2024
    """
    training_dataset = load_data(male_data=male_data)

    # Get unique player names and create mapping
    player_names = np.unique(training_dataset[["Winner", "Loser"]].to_numpy().flatten())
    player_index_map = {x: index for index, x in enumerate(player_names)}
    training_dataset["winner_index"] = training_dataset["Winner"].map(player_index_map)
    training_dataset["loser_index"] = training_dataset["Loser"].map(player_index_map)

    # Initialise the graph edge weights
    edge_values = np.zeros((len(player_index_map), len(player_index_map)))
    edge_weights = np.zeros((len(player_index_map), len(player_index_map)))

    # Split the data into pre-Wimbledon periods.
    wimbledon_filter = (training_dataset["Tournament"] == "Wimbledon").to_numpy()
    wimbledon_change = wimbledon_filter[1:] != wimbledon_filter[:-1]
    training_dataset["period_index"] = np.append(
        np.array([0]), np.cumsum(wimbledon_change)
    )

    # Iterate through the train and test periods
    period_grouped_data = training_dataset.groupby("period_index")

    # Initialise the previous period end date
    previous_period_end_date = date(2019, 1, 1)

    # Initialise overall RMSE
    overall_rmse = 0.0
    for period in range(len(period_grouped_data)):
        period_data = period_grouped_data.get_group(period)

        if period % 2 == 1:
            print(f"Running forecast for period {period}...")
            forecast_objective, baseline_objective = run_wimbledon_forecast(
                period_data, edge_weights, edge_values
            )
            print(
                f"""Wimbledon period {period} with RMSE {forecast_objective}
                    compared to baseline {baseline_objective}"""
            )
            if period >= 7:
                overall_rmse += forecast_objective

        # Calculate the decay in the weight
        max_date = period_data["match_date"].iloc[-1]
        date_difference = (
            max_date - period_data["match_date"]
        ).dt.total_seconds() / 86400
        weight_decay = np.power(temporal_decay, date_difference / 365)

        # Add in surface weights
        weight_decay[period_data["Surface"] == "Hard"] *= hard_weight
        weight_decay[period_data["Surface"] == "Clay"] *= clay_weight

        # Decay the existing data in the weights matrix
        existing_decay_days = (
            max_date - pd.Timestamp(previous_period_end_date)
        ).total_seconds() / 86400
        existing_weight_decay = np.power(temporal_decay, existing_decay_days / 365)

        edge_weights *= existing_weight_decay
        edge_values *= existing_weight_decay

        # Add in the matches from this period
        winner_indices = period_data["winner_index"].to_numpy()
        loser_indices = period_data["loser_index"].to_numpy()
        np.add.at(edge_weights, (winner_indices, loser_indices), weight_decay)
        np.add.at(
            edge_values,
            (winner_indices, loser_indices),
            period_data["additive_win_prob"].to_numpy() * weight_decay,
        )

        # Update the previous weight date
        previous_period_end_date = max_date
    return forecast_objective


def run_wimbledon_forecast(
    results_data: pd.DataFrame, edge_weights: np.ndarray, edge_values: np.ndarray
) -> Tuple[float, float]:
    """
    Run the Wimbledon forecast based on the data

    Args:
        edge_weights (np.ndarray): The edge weights
        edge_values (np.ndarray): The edge values

    Returns:
        float: The RMSE between predicted probabilities and bookies odds
        float: The baseline RMSE
    """
    # Create the player strength mapping
    player_strengths = create_player_strengths(edge_weights, edge_values)
    player_strength_map = dict(enumerate(player_strengths))

    # Create the winner and loser strengths and the estimated probability
    winner_strength = results_data["winner_index"].map(player_strength_map)
    loser_strength = results_data["loser_index"].map(player_strength_map)
    results_data["estimated_probability"] = 1 / (
        1 + np.exp(winner_strength - loser_strength)
    )
    results_data[["Winner", "Loser", "true_win_prob", "estimated_probability"]].to_csv(
        "wimbledon_forecast_results.csv", index=False
    )

    # Create the objective value and baseline
    objective_value = np.sqrt(
        np.mean(
            np.square(
                results_data["estimated_probability"] - results_data["true_win_prob"]
            )
        )
    )
    baseline_value = np.sqrt(
        np.mean(
            np.square(
                results_data["true_win_prob"] - results_data["true_win_prob"].mean()
            )
        )
    )

    return objective_value, baseline_value


def create_player_strengths(
    edge_weights: np.ndarray, edge_values: np.ndarray
) -> np.ndarray:
    """
    Run the Wimbledon forecast based on the data

    Args:
        edge_weights (np.ndarray): The edge weights
        edge_values (np.ndarray): The edge values

    Returns:
        np.ndarray: The player strengths
    """
    # Get average edge value estimate
    average_edge_values = edge_values.copy()
    average_edge_values[edge_weights > 0] = (
        edge_values[edge_weights > 0] / edge_weights[edge_weights > 0]
    )

    def player_ranking_objective(
        player_rankings: np.ndarray, regulariser_size: float = 0.001
    ):
        """
        Define the objective in player ranking.

        We define this to be the sum of squared differences, with an
            extra small regulariser
        """
        expected_pairwise_rankings = (
            player_rankings[:, np.newaxis] - player_rankings[np.newaxis, :]
        )
        objective = np.sum(
            np.square(average_edge_values - expected_pairwise_rankings) * edge_weights
        ) + regulariser_size * np.sum(np.square(player_rankings))
        if np.random.random() < 0.001:
            print(f"Current objective: {objective}")
        return objective

    # Extract optimal strengths
    optimised_results = minimize(
        player_ranking_objective, np.zeros(len(edge_weights)), method="L-BFGS-B"
    )
    print(
        f"""Optimisation terminated with {optimised_results.message}
            and function value {optimised_results.fun}"""
    )
    optimal_strengths = optimised_results.x

    # Fill in missing players with minimum strength
    weight_sums = np.sum(edge_weights, axis=1)
    maximum_true_strength = np.max(optimal_strengths[weight_sums > 0])
    optimal_strengths[weight_sums == 0] = maximum_true_strength

    return optimal_strengths
