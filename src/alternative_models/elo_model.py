import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

from src.alternative_models.base_alt_model import BaseAlternativeModel
from src.alternative_models.model_registry import register_model

@register_model("elo_model")
class EloModel(BaseAlternativeModel):
    """
    ELO rating model for tennis match prediction.
    
    The ELO model maintains a dynamic rating for each player that updates after
    each match. The probability of player i beating player j is:
        P(i beats j) = 1 / (1 + 10^((rating_j - rating_i) / 400))
    
    After each match, ratings are updated based on the result and the expected probability.
    """
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500, 
                 rating_decay: float = 0.0, surface_adjustment: bool = True):
        """
        Initialize the ELO model.
        
        Args:
            k_factor (float): The K-factor determines how much ratings change after each match.
                             Higher values = more volatile ratings. Standard chess uses 32.
            initial_rating (float): Starting rating for new players (1500 is standard)
            rating_decay (float): Optional decay factor to reduce ratings over time for inactive players
            surface_adjustment (bool): Whether to maintain separate ratings for different surfaces
        """
        super().__init__()
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.rating_decay = rating_decay
        self.surface_adjustment = surface_adjustment
        self.player_ratings: Dict[str, float] = {}
        self.player_last_match: Dict[str, datetime] = {}
        
        # For surface-specific ratings
        if surface_adjustment:
            self.surface_ratings: Dict[Tuple[str, str], float] = {}  # (player, surface) -> rating
    
    def _get_rating(self, player: str, surface: str = None, current_date: datetime = None) -> float:
        """
        Get the current ELO rating for a player, with optional surface adjustment.
        
        Args:
            player (str): Player name
            surface (str): Court surface (e.g., 'Hard', 'Clay', 'Grass')
            current_date (datetime): Current date for applying decay
            
        Returns:
            float: Player's ELO rating
        """
        # Apply rating decay if enabled
        if self.rating_decay > 0 and player in self.player_last_match and current_date:
            days_inactive = (current_date - self.player_last_match[player]).days
            decay_amount = days_inactive * self.rating_decay
            
            if self.surface_adjustment and surface:
                key = (player, surface)
                if key in self.surface_ratings:
                    self.surface_ratings[key] = max(
                        self.initial_rating,
                        self.surface_ratings[key] - decay_amount
                    )
            else:
                if player in self.player_ratings:
                    self.player_ratings[player] = max(
                        self.initial_rating,
                        self.player_ratings[player] - decay_amount
                    )
        
        # Return surface-specific or general rating
        if self.surface_adjustment and surface:
            key = (player, surface)
            return self.surface_ratings.get(key, self.initial_rating)
        else:
            return self.player_ratings.get(player, self.initial_rating)
    
    def _set_rating(self, player: str, rating: float, surface: str = None, current_date: datetime = None):
        """
        Set the ELO rating for a player.
        
        Args:
            player (str): Player name
            rating (float): New rating value
            surface (str): Court surface (if using surface adjustment)
            current_date (datetime): Date to update last match time
        """
        if self.surface_adjustment and surface:
            self.surface_ratings[(player, surface)] = rating
        else:
            self.player_ratings[player] = rating
        
        if current_date:
            self.player_last_match[player] = current_date
    
    def _expected_score(self, rating1: float, rating2: float) -> float:
        """
        Calculate the expected score (win probability) for player 1.
        
        Args:
            rating1 (float): ELO rating of player 1
            rating2 (float): ELO rating of player 2
            
        Returns:
            float: Expected probability of player 1 winning
        """
        return 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))
    
    def _update_ratings(self, winner: str, loser: str, surface: str = None, 
                       match_date: datetime = None, k_factor: float = None):
        """
        Update ELO ratings after a match.
        
        Args:
            winner (str): Name of the winning player
            loser (str): Name of the losing player
            surface (str): Court surface
            match_date (datetime): Date of the match
            k_factor (float): Optional override for K-factor (e.g., for tournament importance)
        """
        if k_factor is None:
            k_factor = self.k_factor
        
        # Get current ratings
        winner_rating = self._get_rating(winner, surface, match_date)
        loser_rating = self._get_rating(loser, surface, match_date)
        
        # Calculate expected scores
        winner_expected = self._expected_score(winner_rating, loser_rating)
        loser_expected = self._expected_score(loser_rating, winner_rating)
        
        # Update ratings (winner gets 1, loser gets 0)
        new_winner_rating = winner_rating + k_factor * (1.0 - winner_expected)
        new_loser_rating = loser_rating + k_factor * (0.0 - loser_expected)
        
        # Set new ratings
        self._set_rating(winner, new_winner_rating, surface, match_date)
        self._set_rating(loser, new_loser_rating, surface, match_date)
    
    def _train_model(self, training_data: pd.DataFrame):
        """
        Train the ELO model by processing historical matches chronologically.
        
        Args:
            training_data (pd.DataFrame): Historical match data sorted by date
        """
        # Reset ratings
        self.player_ratings = {}
        self.player_last_match = {}
        if self.surface_adjustment:
            self.surface_ratings = {}
        
        # Process matches in chronological order
        training_data = training_data.sort_values('match_date')
        
        for idx, match in training_data.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            surface = match.get('Surface', None) if self.surface_adjustment else None
            match_date = match['match_date']
            
            # Adjust K-factor for Grand Slams or five-set matches
            k_factor = self.k_factor
            if match.get('five_sets', False):
                k_factor *= 1.2  # Increase importance of Grand Slam matches
            
            # Update ratings
            self._update_ratings(winner, loser, surface, match_date, k_factor)
    
    def predict(self, target_tournament: str, target_year: int, male_data: bool):
        """
        Create model predictions for the selected tournament and year.
        
        Args:
            target_tournament (str): The name of the tournament we are targeting
            target_year (int): The year of the tournament we want to predict
            male_data (bool): Whether or not we want to perform these predictions on male data
        """
        # Load all data
        all_data = self._get_data(male_data)
        
        # Filter for target tournament matches
        target_matches = all_data[
            (all_data['Tournament'] == target_tournament) & 
            (all_data['match_date'].dt.year == target_year)
        ].copy()
        
        if len(target_matches) == 0:
            print(f"No matches found for {target_tournament} in {target_year}")
            return
        
        # Get the earliest date in the target tournament
        tournament_start = target_matches['match_date'].min()
        
        # Get all matches before the tournament for training
        training_data = all_data[all_data['match_date'] < tournament_start].copy()
        
        print(f"Training ELO model on {len(training_data)} matches...")
        print(f"Training period: {training_data['match_date'].min().date()} to {tournament_start.date()}")
        
        # Train the model
        self._train_model(training_data)
        print(f"Trained ratings for {len(self.player_ratings)} players")
        
        # Sort target matches chronologically for sequential prediction
        target_matches = target_matches.sort_values('match_date')
        
        # Make predictions for target tournament
        predictions = []
        
        for idx, match in target_matches.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            surface = match.get('Surface', None)
            match_date = match['match_date']
            
            # Get ratings before the match
            winner_rating = self._get_rating(winner, surface, match_date)
            loser_rating = self._get_rating(loser, surface, match_date)
            
            # Predict probability of winner winning
            prob_winner = self._expected_score(winner_rating, loser_rating)
            
            predictions.append({
                'Date': match['Date'],
                'Tournament': match['Tournament'],
                'Winner': winner,
                'Loser': loser,
                'winner_rating_before': winner_rating,
                'loser_rating_before': loser_rating,
                'predicted_prob_winner': prob_winner,
                'predicted_prob_loser': 1 - prob_winner,
                'predicted_correctly': prob_winner > 0.5
            })
            
        
        # Create predictions DataFrame
        prediction_df = pd.DataFrame(predictions)
        
        # Calculate accuracy
        accuracy = prediction_df['predicted_correctly'].mean()
        avg_prob = prediction_df['predicted_prob_winner'].mean()
        
        print(f"\nPrediction accuracy: {accuracy:.2%}")
        print(f"Average predicted probability for winner: {avg_prob:.2%}")
        print(f"Total matches predicted: {len(prediction_df)}")
        
        # Save predictions
        self._save_predictions(prediction_df, target_tournament, target_year, male_data)
        print(f"Predictions saved to {self.name}/")
        
        return prediction_df
    
    def get_top_players(self, n: int = 10) -> list:
        """
        Get the top N players by ELO rating.
        
        Args:
            n (int): Number of top players to return
            
        Returns:
            list: List of tuples (player_name, rating)
        """
        sorted_players = sorted(self.player_ratings.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        return sorted_players[:n]