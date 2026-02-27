import pandas as pd
from datetime import datetime
from abc import abstractmethod, ABC
import os


class BaseAlternativeModel(ABC):
    def __init__(self):
        """
        Define the base class for alternative models
        """
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def predict(self, target_tournament: str, target_year: int, male_data: bool):
        """
        Create the model predictions for the selected tournament and year

        Args:
            target_tournament (str): The name of the tournament we are targetting
            target_year (int): The year of the tournament we want to predict
            male_data (bool): Whether or not we want to perform these predictions on male
                data
        """

    def _save_predictions(
        self,
        prediction_data: pd.DataFrame,
        target_tournament: str,
        target_year: int,
        male_data: bool,
    ):
        """
        Save the predictions to a CSV

        Args:
            prediction_data (pd.DataFrame): The prediction data with predicted probabilities and
                results in each match
            target_tournament (str): The name of the tournament we are targetting
            target_year (int): The year of the tournament we want to predict
            male_data (bool): Whether or not we want to perform these predictions on male
                data
        """
        # Create a directory if necessary
        if not os.path.isdir(self.name):
            os.mkdir(self.name)

        # Save the data
        male_string = "male" if male_data else "female"
        prediction_data.to_csv(
            f"{self.name}/{male_string}_{target_tournament.lower()}_{target_year}.csv",
            index=False,
        )

    @staticmethod
    def _get_data(male_data: bool) -> pd.DataFrame:
        """
        Load the data into a single pd.DataFrame, containing columns of
            Tournament, Date, Winner, Loser

        Args:
            male_data (bool): Whether we are looking for male data

        Returns:
            pd.DataFrame: The concatenated data
        """
        # Iterate through the saved CSVs
        file_suffix = "men" if male_data else "women"
        loaded_dfs: list[pd.DataFrame] = []
        for year in range(2019, 2026):
            loaded_dfs.append(
                pd.read_csv(f"src/main_package/data/{year}_{file_suffix}.csv")
            )

        # Perform light processing on the data
        combined_df = pd.concat(loaded_dfs)
        combined_df["match_date"] = combined_df["Date"].apply(
            lambda x: datetime.strptime(x, "%m/%d/%Y")
        )

        # Create a five-set filter
        combined_df["five_sets"] = False
        five_set_mens_tournaments = [
            "Australian Open",
            "French Open",
            "Wimbledon",
            "US Open",
        ]
        if male_data:
            combined_df.loc[
                combined_df["Tournament"].isin(five_set_mens_tournaments), "five_sets"
            ] = True

        return combined_df
