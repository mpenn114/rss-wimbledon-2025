import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict
from src.alternative_models.base_alt_model import BaseAlternativeModel
from src.alternative_models.model_registry import register_model

@register_model("bradley_terry_model")
class BradleyTerryModel(BaseAlternativeModel):
    """
    Bradley-Terry model for tennis match prediction.
    
    The Bradley-Terry model estimates a strength parameter for each player
    such that the probability of player i beating player j is:
        P(i beats j) = strength_i / (strength_i + strength_j)
    
    Player strengths are estimated using maximum likelihood estimation
    on historical match data.
    """
    
    def __init__(self, training_window_days: int = 365, regularization: float = 0.01):
        """
        Initialize the Bradley-Terry model.
        
        Args:
            training_window_days (int): Number of days of historical data to use for training
            regularization (float): L2 regularization parameter to prevent overfitting
        """
        super().__init__()
        self.training_window_days = training_window_days
        self.regularization = regularization
        self.player_strengths: Dict[str, float] = {}
        
    def _estimate_strengths(self, match_data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate player strengths using maximum likelihood estimation.
        
        Args:
            match_data (pd.DataFrame): Historical match data with Winner and Loser columns
            
        Returns:
            Dict[str, float]: Dictionary mapping player names to strength parameters
        """
        # Get unique players
        players = list(set(match_data['Winner'].unique()) | set(match_data['Loser'].unique()))
        player_to_idx = {player: idx for idx, player in enumerate(players)}
        n_players = len(players)
        
        # Initialize strengths (log-scale for numerical stability)
        initial_strengths = np.zeros(n_players)

        # Precompute the winner and loser indicies
        winner_indices = np.array([player_to_idx[winner] for winner in match_data['Winner']])
        loser_indices = np.array([player_to_idx[loser] for loser in match_data['Loser']])
        
        def negative_log_likelihood(log_strengths):
            """
            Compute negative log-likelihood for optimization.
            Uses log-scale strengths for numerical stability.
            """
            # Calculate strengths
            strengths = np.exp(log_strengths)
            
            # Bradley-Terry probability: P(i beats j) = s_i / (s_i + s_j)
            prob_win = strengths[winner_indices] / (strengths[winner_indices] + strengths[loser_indices])
            
            # Avoid log(0)
            prob_win = np.clip(prob_win, 1e-10, 1 - 1e-10)
            
            # Add to negative log-likelihood
            nll = (-1)*np.sum(np.log(prob_win))
            
            # Add L2 regularization to prevent extreme values
            nll += self.regularization * np.sum(log_strengths ** 2)
            
            return nll
        
        # Optimize using L-BFGS-B
        result = minimize(
            negative_log_likelihood,
            initial_strengths,
            method='L-BFGS-B',
        )
        
        # Convert log-strengths back to strengths
        optimized_log_strengths = result.x
        optimized_strengths = np.exp(optimized_log_strengths)
        
        # Normalize strengths to have mean of 1.0 for interpretability
        optimized_strengths = optimized_strengths / np.mean(optimized_strengths)
        
        # Create dictionary mapping players to strengths
        strength_dict = {player: optimized_strengths[idx] 
                        for player, idx in player_to_idx.items()}
        
        
        return strength_dict
    
    def _predict_match(self, player1: str, player2: str) -> float:
        """
        Predict the probability of player1 beating player2.
        
        Args:
            player1 (str): Name of first player
            player2 (str): Name of second player
            
        Returns:
            float: Probability of player1 winning (between 0 and 1)
        """
        # Get strengths, default to 1.0 for unknown players
        strength1 = self.player_strengths.get(player1, 1.0)
        strength2 = self.player_strengths.get(player2, 1.0)
        
        # Bradley-Terry probability formula
        prob_player1_wins = strength1 / (strength1 + strength2)
        
        return prob_player1_wins
    
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
        
        # Get training data: all matches before tournament start within the training window
        training_cutoff = tournament_start - pd.Timedelta(days=self.training_window_days)
        training_data = all_data[
            (all_data['match_date'] >= training_cutoff) & 
            (all_data['match_date'] < tournament_start)
        ]
        
        print(f"Training Bradley-Terry model on {len(training_data)} matches...")
        print(f"Training period: {training_cutoff.date()} to {tournament_start.date()}")
        
        # Estimate player strengths from training data
        self.player_strengths = self._estimate_strengths(training_data)
        print(f"Estimated strengths for {len(self.player_strengths)} players")
        
        # Make predictions for target tournament
        predictions = []
        
        for idx, match in target_matches.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            
            # Predict probability of winner winning
            prob_winner = self._predict_match(winner, loser)
            
            predictions.append({
                'Date': match['Date'],
                'Tournament': match['Tournament'],
                'Winner': winner,
                'Loser': loser,
                'predicted_prob_winner': prob_winner,
                'predicted_prob_loser': 1 - prob_winner,
                'predicted_correctly': prob_winner > 0.5
            })
        
        # Create predictions DataFrame
        prediction_df = pd.DataFrame(predictions)
        
        # Calculate accuracy
        accuracy = prediction_df['predicted_correctly'].mean()
        print(f"\nPrediction accuracy: {accuracy:.2%}")
        print(f"Total matches predicted: {len(prediction_df)}")
        
        # Save predictions
        self._save_predictions(prediction_df, target_tournament, target_year, male_data)
        print(f"Predictions saved to {self.name}/")
        
        return prediction_df