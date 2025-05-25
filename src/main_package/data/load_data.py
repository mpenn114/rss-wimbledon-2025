import pandas as pd
from typing import List


def load_data(male_data: bool) -> pd.DataFrame:
    """
    Load the data into a single pd.DataFrame

    Args:
        male_data (bool): Whether we are looking for male data

    Returns:
        pd.DataFrame: The concatenated data
    """
    file_suffix = "men" if male_data else "women"
    loaded_dfs: List[pd.DataFrame] = []
    for year in range(2019, 2025):
        loaded_dfs.append(
            pd.read_csv(f"src/main_package/data/{year}_{file_suffix}.csv")
        )

    return pd.concat(loaded_dfs)
