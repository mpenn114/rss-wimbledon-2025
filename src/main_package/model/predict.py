import pandas as pd
import numpy as np
from .utils import define_prize_money
from src.main_package.model.train import train_model
def predict_wimbledon_prize_money(male_data:bool) -> pd.DataFrame:
    """
    Predict the prize money that each player will get in Wimbledon

    Args:
        male_data (bool): Whether or not we want male data
        num_simulations (int): The number of simulations to do
    """
    # Get the player strength map
    _, player_strength_map = train_model(
        temporal_decay=0.15,
        grass_weight=5.0,
        male_data = male_data,
        return_player_strengths=True
    )

    # Load the draw. We expect the CSV to contain players in the same order as the tree
    tournament_draw = load_demo_draw(male_data)
    player_strengths = tournament_draw['player_name'].map(player_strength_map).to_numpy()
    
    # Calculate the probability of each player beating each other player
    win_probabilities = 1 / (
        1 + np.exp(player_strengths[:,np.newaxis] - player_strengths[np.newaxis,:])
    )

    # Calculate the progression probabilities
    progression = np.zeros((9,128))
    progression[0] = 1

    for round_index in range(1,9):
        for player in range(128):
            # Find the block that the player is playing against
            block_size = 2**(round_index-1)
            block_index = player // block_size
            if block_index % 2 == 0:
                block_index += 1
            else:
                block_index -= 1
           
            # Calculate the probability of that player reaching the next round
            progression[round_index,player] = progression[round_index-1, player]*np.sum(
                win_probabilities[player, block_size*block_index:block_size*(block_index + 1)]
                * progression[round_index-1, block_size*block_index:block_size*(block_index + 1)]
                )
    # Calculate prize money
    round_reached_probability= progression[:-1] - progression[1:]
    prize_money = define_prize_money()
    tournament_draw['mean_prize_money'] = np.sum(round_reached_probability*prize_money[:,np.newaxis],axis=0)
    return tournament_draw

def load_demo_draw(male_data:bool):
    """
    Load a demo version of the draw for testing

    Args:
        male_data (bool): Whether or not we are using the male data
    """
    results_df = pd.read_csv(f"src/main_package/data/2024_{'men' if male_data else 'women'}.csv")
    wimbledon_first_round = results_df[results_df['Tournament'] == 'Wimbledon'].reset_index(drop=True).iloc[:64]
    wimbledon_single_column = wimbledon_first_round[['Winner','Loser']].to_numpy().flatten()
    return pd.DataFrame(wimbledon_single_column, columns = ['player_name'])
