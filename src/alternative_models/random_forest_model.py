import pandas as pd
import numpy as np
from typing import Optional, Dict
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

from src.alternative_models.base_alt_model import BaseAlternativeModel
from src.alternative_models.model_registry import register_model


@register_model("random_forest_model")
class CatBoostTennisModel(BaseAlternativeModel):
    """
    Simple baseline CatBoost model for tennis match prediction.
    Uses only available features without estimation.

    Available columns:
    WTA, Location, Tournament, Date, Tier, Court, Surface, Round, Best of,
    Winner, Loser, WRank, LRank, WPts, LPts, W1, L1, W2, L2, W3, L3, Wsets, Lsets
    """

    def __init__(self, catboost_params: Optional[Dict] = None):
        """
        Initialize the CatBoost Tennis Model.

        Args:
            catboost_params: CatBoost hyperparameters (uses defaults if None)
        """
        super().__init__()

        # Default CatBoost parameters
        self.catboost_params = catboost_params or {
            'depth': 6,
            'learning_rate': 0.05,
            'iterations': 500,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': 42,
            'verbose': False,
            "early_stopping_rounds":50
        }

        self.model = None
        self.feature_columns = None

    def _calculate_player_statistics(self, data: pd.DataFrame, player_name: str, cutoff_date: pd.Timestamp) -> Dict:
        """
        Calculate rolling statistics for a specific player up to (but not including) the cutoff date.
        
        Args:
            data: Historical match data
            player_name: Name of the player
            cutoff_date: Only use matches before this date
            
        Returns:
            Dictionary of player statistics
        """
        # Get matches from last 12 months before cutoff (but not including cutoff date)
        twelve_months_ago = cutoff_date - pd.Timedelta(days=365)
        recent_data = data[(data['match_date'] >= twelve_months_ago) & 
                          (data['match_date'] < cutoff_date)].copy()
        
        # Matches where player participated
        won_matches = recent_data[recent_data['Winner'] == player_name]
        lost_matches = recent_data[recent_data['Loser'] == player_name]
        
        total_matches = len(won_matches) + len(lost_matches)
        
        # Initialize default stats
        stats = {
            'points_pct_12m': 0.0,
            'avg_rank_when_winner_12m': 9999.0,
            'avg_rank_when_loser_12m': 9999.0
        }
        
        if total_matches > 0:
            # Points percentage (assuming 2 points for win, 0 for loss)
            points_won = len(won_matches) * 2
            total_possible_points = total_matches * 2
            stats['points_pct_12m'] = points_won / total_possible_points if total_possible_points > 0 else 0.0
            
            # Average rank when winner
            if len(won_matches) > 0:
                winner_ranks = won_matches['WRank'].dropna()
                if len(winner_ranks) > 0:
                    stats['avg_rank_when_winner_12m'] = winner_ranks.mean()
            
            # Average rank when loser
            if len(lost_matches) > 0:
                loser_ranks = lost_matches['LRank'].dropna()
                if len(loser_ranks) > 0:
                    stats['avg_rank_when_loser_12m'] = loser_ranks.mean()
        
        return stats

    def _extract_features(self, row: pd.Series, player1_is_winner: bool, 
                         all_data: pd.DataFrame, match_date: pd.Timestamp) -> Dict:
        """
        Extract features from a match row.

        Args:
            row: Match data row
            player1_is_winner: True if player1 is the winner, False if loser
            all_data: All historical data (for calculating rolling stats)
            match_date: Date of the current match (to avoid data leakage)

        Returns:
            Dictionary of features
        """
        features = {}

        # Determine player names
        if player1_is_winner:
            player1_name = row['Winner']
            player2_name = row['Loser']
        else:
            player1_name = row['Loser']
            player2_name = row['Winner']

        # Ranking features (lower is better)
        w_rank = row.get('WRank', 9999)
        l_rank = row.get('LRank', 9999)

        if player1_is_winner:
            features['player1_rank'] = w_rank if pd.notna(w_rank) else 9999
            features['player2_rank'] = l_rank if pd.notna(l_rank) else 9999
        else:
            features['player1_rank'] = l_rank if pd.notna(l_rank) else 9999
            features['player2_rank'] = w_rank if pd.notna(w_rank) else 9999

        # Calculate rolling statistics for each player individually
        player1_stats = self._calculate_player_statistics(all_data, player1_name, match_date)
        player2_stats = self._calculate_player_statistics(all_data, player2_name, match_date)
        
        features['player1_points_pct_12m'] = player1_stats['points_pct_12m']
        features['player2_points_pct_12m'] = player2_stats['points_pct_12m']
        features['player1_avg_rank_winner_12m'] = player1_stats['avg_rank_when_winner_12m']
        features['player2_avg_rank_winner_12m'] = player2_stats['avg_rank_when_winner_12m']
        features['player1_avg_rank_loser_12m'] = player1_stats['avg_rank_when_loser_12m']
        features['player2_avg_rank_loser_12m'] = player2_stats['avg_rank_when_loser_12m']

        # Surface encoding (one-hot)
        surface = row.get('Surface', 'Hard')
        features['surface_hard'] = 1 if surface == 'Hard' else 0
        features['surface_clay'] = 1 if surface == 'Clay' else 0
        features['surface_grass'] = 1 if surface == 'Grass' else 0

        # Tournament tier (if available)
        tier = row.get('Tier', '')
        features['is_grand_slam'] = 1 if 'Grand Slam' in str(tier) else 0

        # Best of (3 or 5 sets)
        best_of = row.get('Best of', 3)
        features['best_of_5'] = 1 if best_of == 5 else 0

        # Round encoding (simple numeric representation)
        round_name = row.get('Round', '')
        round_mapping = {
            'R128': 1, '1st Round': 1,
            'R64': 2, '2nd Round': 2,
            'R32': 3, '3rd Round': 3,
            'R16': 4, '4th Round': 4,
            'Quarterfinals': 5, 'QF': 5,
            'Semifinals': 6, 'SF': 6,
            'Final': 7, 'F': 7
        }
        features['round_num'] = round_mapping.get(round_name, 0)

        return features

    def _prepare_training_data(self, training_data: pd.DataFrame) -> tuple:
        """
        Prepare training data for CatBoost.

        Args:
            training_data: Historical match data

        Returns:
            X: Feature matrix
            y: Labels (1 if player1 won, 0 otherwise)
        """
        features_list = []
        labels = []

        for idx, row in training_data.iterrows():
            match_date = row['match_date']
            
            # Randomly assign winner/loser to player1/player2 to avoid positional bias
            player1_is_winner = np.random.random() > 0.5

            # Pass all training data and match date for calculating rolling stats
            features = self._extract_features(row, player1_is_winner, training_data, match_date)
            features_list.append(features)
            labels.append(1 if player1_is_winner else 0)

        X = pd.DataFrame(features_list)
        y = pd.Series(labels)

        return X, y

    def predict(self, target_tournament: str, target_year: int, male_data: bool, val_perc:float = 0.1):
        """
        Create model predictions for the selected tournament and year.

        Args:
            target_tournament (str): The name of the tournament we are targeting
            target_year (int): The year of the tournament we want to predict
            male_data (bool): Whether or not we want to perform these predictions on male data
            val_perc (float): The percentage of data to validate on in training
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
            return None

        # Get tournament start date
        tournament_start = target_matches['match_date'].min()

        # Training data: all matches before tournament
        training_data = all_data[all_data['match_date'] < tournament_start].copy()

        print(f"Training CatBoost model on {len(training_data)} matches...")
        print(f"Training period: {training_data['match_date'].min().date()} to {tournament_start.date()}")

        # Prepare training data
        X_train, y_train = self._prepare_training_data(training_data)

        # Split into train and val
        val_filter = np.random.random(size=len(X_train)) < val_perc
        self.feature_columns = X_train.columns

        print(f"Extracted {X_train.shape[1]} features")
        print(f"Features: {list(X_train.columns)}")

        # Train CatBoost model
        print("\nTraining CatBoost model...")
        self.model = CatBoostClassifier(**self.catboost_params)
        self.model.fit(X_train[~val_filter], y_train[~val_filter],eval_set=(X_train[val_filter],y_train[val_filter]),
                       use_best_model=True)

        # Make predictions on target tournament
        predictions = []

        for _, match in target_matches.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            match_date = match['match_date']

            # Extract features (winner as player1)
            # Pass all_data (up to tournament) and match_date for calculating rolling stats
            features = self._extract_features(match, player1_is_winner=True, 
                                            all_data=all_data, match_date=match_date)
            X_pred = pd.DataFrame([features])

            # Ensure columns match training data
            X_pred = X_pred[self.feature_columns]

            # Predict probability that player1 (winner) wins
            prob_winner = self.model.predict_proba(X_pred)[0, 1]

            predictions.append({
                'Date': match['Date'],
                'Tournament': match['Tournament'],
                'Winner': winner,
                'Loser': loser,
                'Surface': match.get('Surface', ''),
                'Round': match.get('Round', ''),
                'WRank': match.get('WRank', None),
                'LRank': match.get('LRank', None),
                'predicted_prob_winner': prob_winner,
                'predicted_prob_loser': 1 - prob_winner,
                'predicted_correctly': prob_winner > 0.5
            })

        # Create predictions DataFrame
        prediction_df = pd.DataFrame(predictions)

        # Calculate metrics
        accuracy = prediction_df['predicted_correctly'].mean()
        avg_prob = prediction_df['predicted_prob_winner'].mean()

        # Calculate log loss and Brier score
        y_true = np.ones(len(prediction_df))  # All are winners
        y_pred = prediction_df['predicted_prob_winner'].values

        logloss = log_loss(y_true, y_pred, labels=[0,1])
        brier = brier_score_loss(y_true, y_pred,)

        # Calculate ROC AUC (need to include both winners and losers)
        y_true_full = np.concatenate([
            np.ones(len(prediction_df)),
            np.zeros(len(prediction_df))
        ])
        y_pred_full = np.concatenate([
            prediction_df['predicted_prob_winner'].values,
            prediction_df['predicted_prob_loser'].values
        ])
        roc_auc = roc_auc_score(y_true_full, y_pred_full)

        print(f"\n{'=' * 60}")
        print(f"Model Performance on {target_tournament} {target_year}")
        print(f"{'=' * 60}")
        print(f"Accuracy:                    {accuracy:.2%}")
        print(f"ROC AUC:                     {roc_auc:.4f}")
        print(f"Log Loss:                    {logloss:.4f}")
        print(f"Brier Score:                 {brier:.4f}")
        print(f"Avg predicted prob (winner): {avg_prob:.2%}")
        print(f"Total matches:               {len(prediction_df)}")

        # Feature importance
        if hasattr(self.model, 'get_feature_importance'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.get_feature_importance()
            }).sort_values('importance', ascending=False)

            print(f"\n{'=' * 60}")
            print("Feature Importance")
            print(f"{'=' * 60}")
            for _, row in importance_df.iterrows():
                print(f"{row['feature']:25s} {row['importance']:.4f}")

        # Save predictions
        self._save_predictions(prediction_df, target_tournament, target_year, male_data)
        print(f"\nPredictions saved to {self.name}/")

        return prediction_df