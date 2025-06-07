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
    combined_df = create_estimated_true_probs(combined_df, male_data)

    # Drop NA on various values
    combined_df = combined_df.dropna(
        subset=["Winner", "Loser", "Surface", "Tournament", "true_win_prob"]
    )

    # Return the dataframe sorted by match date
    return combined_df.sort_values(by="match_date").reset_index(drop=True)


def create_estimated_true_probs(combined_df: pd.DataFrame, male_data:bool) -> pd.DataFrame:
    """
    Create the estimated true probabilities for results

    Args:
        combined_df (pd.DataFrame): The combined data
        male_data (bool): Whether or not we have male data

    Returns:
        pd.DataFrame: The original dataframe with a probability for true_win_prob
    """
    five_set_mens_tournaments = [
        'Australian Open',
        'French Open',
        'Wimbledon',
        'US Open'
    ]
    # We adjust the odds to be five-set specific if it is male data
    # Bin(3,q) >=2 = p and we want Bin(5,q) >= 3
    # q^3 + 3q^2(1-q) = p
    three_set_grid_q = np.arange(10_000)/9999
    three_set_grid_prob = np.power(three_set_grid_q,3) + 3*np.power(three_set_grid_q,2)

    if male_data:
        # Adjust the probabilities to be for five sets
        three_set_filter = ~combined_df['Tournament'].isin(five_set_mens_tournaments)
        # Change the winner probability
        set_win_probs = np.interp(combined_df.loc[three_set_filter,'AvgW'].to_numpy().astype(float), three_set_grid_prob, three_set_grid_q)
        five_set_win_probs = np.power(set_win_probs,5) + 5*np.power(set_win_probs,4) + 10*np.power(set_win_probs,3)
        combined_df.loc[three_set_filter,'AvgW'] = five_set_win_probs
        # Change the loser probability
        set_loss_probs = np.interp((1-combined_df.loc[three_set_filter,'AvgL'].to_numpy().astype(float)), three_set_grid_prob, three_set_grid_q)
        five_set_loss_probs = np.power(set_loss_probs,5) + 5*np.power(set_loss_probs,4) + 10*np.power(set_loss_probs,3)
        combined_df.loc[three_set_filter,'AvgL'] = 1-five_set_loss_probs
        

    combined_df["true_win_prob"] = (1 / combined_df["AvgW"]).to_numpy() / (
        (1 / combined_df[["AvgW", "AvgL"]].to_numpy()).sum(axis=1)
    )

    combined_df["additive_win_prob"] = np.log(
        (1 - combined_df["true_win_prob"].to_numpy())
        / combined_df["true_win_prob"].to_numpy()
    )

    return combined_df
