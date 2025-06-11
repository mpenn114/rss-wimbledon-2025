import pandas as pd
import numpy as np
from src.main_package.model.train import train_model
def predict_wimbledon_prize_money(male_data:bool):
    """
    Predict the prize money that each player will get in Wimbledon

    Args:
        male_data (bool): Whether or not we want male data
        num_simulations (int): The number of simulations to do
    """
    # Get the player strength map
    _, player_strength_map = train_model(
        temporal_decay=0.4,
        clay_weight=0.8,
        hard_weight=0.9,
        male_data = male_data,
        return_player_strengths=True
    )

    # Load the draw. We expect the CSV to contain players in the same order as the tree
    tournament_draw = load_demo_draw()
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
            print(win_probabilities[player, block_size*block_index:block_size*(block_index + 1)])
            print(block_size)

            print('___________')
            progression[round_index,player] = progression[round_index-1, player]*np.sum(
                win_probabilities[player, block_size*block_index:block_size*(block_index + 1)]
                * progression[round_index-1, block_size*block_index:block_size*(block_index + 1)]
                )
    # Calculate prize money
    round_reached_probability= progression[:-1] - progression[1:]
    prize_money = np.array([
        60_000,
        93_000,
        143_000,
        226_000,
        375_000,
        715_000,
        1_400_000,
        2_700_000
    ])
    print(round_reached_probability)
    tournament_draw['mean_prize_money'] = np.sum(round_reached_probability*prize_money[:,np.newaxis],axis=0)
    suffix = 'male' if male_data else 'female'
    tournament_draw.to_csv(f"prize_money_{suffix}.csv",index=False)

def load_demo_draw():
    """
    Load a demo version of the draw for testing
    """
    results_df = pd.read_csv('src/main_package/data/2024_men.csv')
    wimbledon_first_round = results_df[results_df['Tournament'] == 'Wimbledon'].reset_index(drop=True).iloc[:64]
    wimbledon_single_column = wimbledon_first_round['Winner'].tolist() + wimbledon_first_round['Loser'].tolist()
    return pd.DataFrame(wimbledon_single_column, columns = ['player_name'])
