import pandas as pd
from typing import List
from datetime import datetime
import numpy as np


def load_data(male_data: bool) -> pd.DataFrame:
    """
    Load the data into a single pd.DataFrame

    Args:
        male_data (bool): Whether we are looking for male data

    Returns:
        pd.DataFrame: The concatenated data
    """
    # Iterate through the saved CSVs
    file_suffix = "men" if male_data else "women"
    loaded_dfs: List[pd.DataFrame] = []
    for year in range(2019, 2025):
        loaded_dfs.append(
            pd.read_csv(f"src/main_package/data/{year}_{file_suffix}.csv")
        )

    # Perform light processing on the data
    combined_df = pd.concat(loaded_dfs)
    combined_df["match_date"] = combined_df["Date"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%Y")
    )
    combined_df = create_estimated_true_probs(combined_df)

    # Drop NA on various values
    combined_df = combined_df.dropna(
        subset=["Winner", "Loser", "Surface", "Tournament", "true_win_prob"]
    )

    # Return the dataframe sorted by match date
    return combined_df.sort_values(by="match_date").reset_index(drop=True)


def create_estimated_true_probs(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the estimated true probabilities for results

    Args:
        combined_df (pd.DataFrame): The combined data

    Returns:
        pd.DataFrame: The original dataframe with a probability for true_win_prob
    """
    combined_df["true_win_prob"] = (1 / combined_df["AvgW"]).to_numpy() / (
        (1 / combined_df[["AvgW", "AvgL"]].to_numpy()).sum(axis=1)
    )

    combined_df["additive_win_prob"] = np.log(
        (1 - combined_df["true_win_prob"].to_numpy())
        / combined_df["true_win_prob"].to_numpy()
    )

    return combined_df
