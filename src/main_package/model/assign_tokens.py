import pandas as pd
import numpy as np
from .utils import define_prize_money
from typing import Tuple, List
from scipy.optimize import minimize
from tqdm import tqdm


def assign_tokens(expected_prize_money: pd.DataFrame, lucky_losers: List[str] = []):
    """
    Assign tokens to the players based on our model assumptions.
    """
    # Set seed
    np.random.seed(42)

    # Get seedings and create seeding-based prize money
    seedings = load_seedings()
    seeding_prize_money = get_seeding_based_expected_winnings(seedings)

    overall_expectation_data = pd.concat(
        [
            seeding_prize_money,
            expected_prize_money.set_index("player_name")["mean_prize_money"],
        ],
        axis=1,
    ).reset_index()
    overall_expectation_data.columns = [
        "player_name",
        "seeding_based_money",
        "mean_prize_money",
    ]

    # Get the alpha distribution
    alpha_grid, optimal_alpha_pdf, _ = calculate_alpha_distribution(
        overall_expectation_data["mean_prize_money"].to_numpy()
    )

    # Fill in mean prize money with average 1st-v-2nd round loser winnings
    prize_money = define_prize_money()
    overall_expectation_data["seeding_based_money"] = overall_expectation_data[
        "seeding_based_money"
    ].fillna(0.5 * (prize_money[0] + prize_money[1]))

    # Store the different tokens assigned to each player
    sampled_tokens = np.zeros(len(overall_expectation_data))

    for _ in tqdm(range(2), desc="Assigning Tokens...."):
        token_df, token_assignment = create_token_assignments(
            overall_expectation_data.copy(), alpha_grid, optimal_alpha_pdf
        )
        token_df.to_csv("token_assignment.csv", index=False)

        def optimise_strategy(tokens: np.ndarray):
            tokens = 10 * np.abs(tokens) / np.sum(np.abs(tokens))
            total_tokens = tokens + token_df["tokens"].to_numpy()
            expected_winnings = np.sum(
                token_df["mean_prize_money"].to_numpy() * tokens / total_tokens
            )
            opponent_winnings = (
                token_df["mean_prize_money"].to_numpy()[:, np.newaxis]
                * token_assignment
                / (total_tokens[:, np.newaxis])
            )
            return -expected_winnings + np.max(np.sum(opponent_winnings, axis=0))

        optimal_solution = minimize(
            optimise_strategy,
            10 * np.ones(256) / 256,
            method="L-BFGS-B",
            options={
                "maxiter": 100_000,  # or any large number
                "maxfun": 100_000,  # optional, can be set if needed
                "disp": True,  # optional, shows convergence messages
            },
        )

        print(f"Optimal Solution Win Margin: {-optimal_solution.fun}")
        tokens = 10 * np.abs(optimal_solution.x) / np.sum(np.abs(optimal_solution.x))
        sampled_tokens += tokens

        total_tokens = tokens + token_df["tokens"].to_numpy()
        expected_winnings = np.sum(
            token_df["mean_prize_money"].to_numpy() * tokens / total_tokens
        )
        opponent_winnings = (
            token_df["mean_prize_money"].to_numpy()[:, np.newaxis]
            * token_assignment
            / (total_tokens[:, np.newaxis])
        )
        print("Opponent Winnings:")
        print(np.sum(opponent_winnings, axis=0))
        print("Our Winnings:")
        print(expected_winnings)

    overall_expectation_data["our_tokens"] = (
        10 * sampled_tokens / np.sum(sampled_tokens)
    )

    # Ensure we place at least 1e-4 on each player
    overall_expectation_data["our_tokens"] = np.clip(
        overall_expectation_data["our_tokens"], 1e-4, 10
    )

    # Add in the lucky losers
    lucky_loser_dataframe = pd.DataFrame(
        {"player_name": lucky_losers, "our_tokens": [1e-7] * len(lucky_losers)}
    )

    overall_expectation_data = pd.concat(
        [overall_expectation_data, lucky_loser_dataframe]
    )

    # Final normalisation
    overall_expectation_data["our_tokens"] = (
        overall_expectation_data["our_tokens"]
        * 10
        / np.sum(overall_expectation_data["our_tokens"])
    )

    # Save
    overall_expectation_data.sort_values(by="our_tokens", ascending=False).to_csv(
        "final_token_selection.csv", index=False
    )


def get_seeding_based_expected_winnings(seeding: pd.Series) -> pd.Series:
    """
    Get seeding-based expected winnings
    """
    seeding_round_reached = np.floor(7 - np.log2(np.clip(seeding, 1, 127))).astype(int)
    prize_money = define_prize_money()
    return pd.Series(
        {
            player: prize_money[round]
            for player, round in zip(seeding.index, seeding_round_reached)
        }
    )


def load_seedings() -> pd.Series:
    """
    Load the player seedings.

    Returns:
        pd.Series: with index equal to the player name and value equal to their seeding
    """
    male_seedings = pd.Series(
        [
            "Sinner J.",
            "Alcaraz C.",
            "Zverev A.",
            "Draper J.",
            "Fritz T.",
            "Djokovic N.",
            "Musetti L.",
            "Rune H.",
            "Medvedev D.",
            "Shelton B.",
            "De Minaur A.",
            "Tiafoe F.",
            "Paul T.",
            "Rublev A.",
            "Mensik J.",
            "Cerundolo F.",
            "Khachanov K.",
            "Humbert U.",
            "Dimitrov G.",
            "Popyrin A.",
            "Machac T.",
            "Cobolli F.",
            "Lehecka J.",
            "Tsitsipas S.",
            "Auger-Aliassime F.",
            "Davidovich Fokina A.",
            "Shapovalov D.",
            "Bublik A.",
            "Nakashima B.",
            "Michelsen A.",
            "Griekspoor T.",
            "Berrettini M.",
        ],
        index=range(1, 33),
        name="2025 Wimbledon Men's Seeds",
    )

    womens_seedings = pd.Series(
        [
            "Sabalenka A.",
            "Gauff C.",
            "Pegula J.",
            "Paolini J.",
            "Zheng Q.",
            "Keys M.",
            "Andreeva M.",
            "Swiatek I.",
            "Navarro E.",
            "Badosa P.",
            "Rybakina E.",
            "Shnaider D.",
            "Anisimova A.",
            "Svitolina E.",
            "Muchova K.",
            "Kasatkina D.",
            "Krejcikova B.",
            "Alexandrova E.",
            "Samsonova L.",
            "Ostapenko J.",
            "Haddad Maia B.",
            "Vekic D.",
            "Tauson C.",
            "Mertens E.",
            "Frech M.",
            "Kostyuk M.",
            "Linette M.",
            "Kenin S.",
            "Fernandez L.",
            "Noskova L.",
            "Krueger A.",
            "Kessler M.",
        ],
        index=range(1, 33),
        name="2025 Wimbledon Women's Seeds",
    )

    # Flip both series to be name -> seeding
    male_seedings_flipped = pd.Series(
        male_seedings.index.values, index=male_seedings.values
    )
    womens_seedings_flipped = pd.Series(
        womens_seedings.index.values, index=womens_seedings.values
    )

    # Concatenate both into a single Series
    all_seedings = pd.concat([male_seedings_flipped, womens_seedings_flipped])
    all_seedings.name = "Seeding"

    return all_seedings


def create_token_assignments(
    overall_expectation_data: pd.DataFrame,
    alpha_grid: np.ndarray,
    optimal_alpha_pdf: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create the token assignments from the random players

    Args:
        overall_expectation_data (pd.Dataframe): The dataframe containing true and
            seedings-based expected winnings
        alpha_grid (np.ndarray): The grid of alpha values
        optimal_alpha_pdf (np.ndarray): The optimal alpha pdf
    """
    # Sample number of competitors
    num_competitors = np.random.poisson(lam=20)

    # Calculate the distribution of alphas
    alpha = np.random.choice(alpha_grid, p=optimal_alpha_pdf, size=num_competitors)

    # Sample expectations, allowing for bias from bookmaker's odds
    true_expectation = overall_expectation_data["mean_prize_money"].to_numpy()
    bookies_expectation = overall_expectation_data["seeding_based_money"].to_numpy()
    random_weights = np.random.random(size=num_competitors)
    player_expectations = true_expectation[:, np.newaxis] * random_weights[
        np.newaxis, :
    ] + bookies_expectation[:, np.newaxis] * (1 - random_weights[np.newaxis, :])
    player_expectations = player_expectations / np.sum(player_expectations)

    # Sample player token assignments
    token_assignment = assign_player_token_weight(
        player_expectations, alpha[np.newaxis, :]
    )
    token_assignment = (
        10 * token_assignment / (np.sum(token_assignment, axis=0)[np.newaxis, :])
    )

    overall_expectation_data["tokens"] = np.sum(token_assignment, axis=1)
    return overall_expectation_data, token_assignment


def calculate_alpha_distribution(
    players_winnings: np.ndarray,
    alpha_max: int = 1.0,
    alpha_min: int = -1.0,
    grid_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the alpha distribution

    We aim to calculate the distribution of alpha such that every alpha has the
      same expected prize money
    """
    # Normalise player winnings
    players_winnings_normalised = players_winnings / np.sum(players_winnings)

    # Get the grid of alpha
    alpha_grid = np.linspace(alpha_min, alpha_max, grid_size)

    # Un-normalised assignments
    total_assigned = assign_player_token_weight(
        players_winnings_normalised[np.newaxis, :], alpha_grid[:, np.newaxis]
    )

    # Get constants
    normalisation_constants = np.sum(total_assigned, axis=1)

    # Get normalised assignments
    total_normalised_assigned = total_assigned / normalisation_constants[:, np.newaxis]

    def objective(alpha_pdf: np.ndarray):
        """
        Define the objective function to minimize
        """
        # Normalise pdf
        alpha_pdf = np.abs(alpha_pdf) / np.sum(np.abs(alpha_pdf))
        # Calculate the total wininngs from each alpha
        total_tokens_assigned = np.sum(
            total_normalised_assigned * alpha_pdf[:, np.newaxis], axis=0
        )
        total_winnings = np.sum(
            total_normalised_assigned
            * players_winnings_normalised[np.newaxis, :]
            / total_tokens_assigned[np.newaxis, :],
            axis=1,
        )

        # Calculate the variation
        variation = np.std(total_winnings)
        return variation

    # Get optimal pdf
    optimised_result = minimize(
        objective,
        np.ones_like(alpha_grid),
        method="L-BFGS-B",
        options={
            "maxiter": 100_000,  # or any large number
            "maxfun": 100_000,  # optional, can be set if needed
            "disp": True,  # optional, shows convergence messages
        },
    )
    print(f"Alpha optimisation terminated with message: {optimised_result.message}")
    optimal_pdf = np.abs(optimised_result.x) / np.sum(np.abs(optimised_result.x))

    # Calculate the total wininngs from each alpha
    total_tokens_assigned = np.sum(
        total_normalised_assigned * optimal_pdf[:, np.newaxis], axis=0
    )
    total_winnings = np.sum(
        total_normalised_assigned
        * players_winnings_normalised[np.newaxis, :]
        / total_tokens_assigned[np.newaxis, :],
        axis=1,
    )

    return alpha_grid, optimal_pdf, total_winnings


def assign_player_token_weight(expected_winnings: np.ndarray, alpha: np.ndarray):
    """
    Assign the number of tokens to a given player

    Args:
        expected_winnings (np.ndarray): The expected winnings from a player
        alpha (np.ndarray): The tuning parameter alpha
    """
    return (alpha >= 0) * (
        (1 - alpha) * expected_winnings + alpha * np.power(expected_winnings, 5)
    ) + (alpha < 0) * (
        (1 + alpha) * expected_winnings - alpha * np.ones_like(expected_winnings)
    )
