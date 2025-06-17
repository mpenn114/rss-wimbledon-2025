import pandas as pd
import numpy as np
from .utils import define_prize_money
from typing import Tuple, List
from scipy.optimize import minimize
def assign_tokens(expected_prize_money:pd.DataFrame, lucky_losers:List[str] = []):
    """
    Assign tokens to the players based on our model assumptions.
    """
    # Set seed
    np.random.seed(42)

    # Get rankings and create ranking-based prize money
    rankings = load_demo_rankings()
    ranking_prize_money = get_ranking_based_expected_winnings(rankings)

    overall_expectation_data = pd.concat([ranking_prize_money, expected_prize_money], axis=1).T
    overall_expectation_data.columns = ['mean_prize_money','ranking_based_money']
    sampled_tokens = np.zeros(len(overall_expectation_data))

    for _ in range(100):

        token_df, token_assignment = create_token_assignments(overall_expectation_data)

        def optimise_strategy(tokens:np.ndarray):
            tokens = 10*np.abs(tokens)/np.sum(np.abs(tokens))
            total_tokens = (tokens + token_df['tokens'].to_numpy())
            expected_winnings = np.sum(
                token_df['expected_winnings'].to_numpy()*tokens/total_tokens
            )
            opponent_winnings = token_df['expected_winnings'].to_numpy()[:,np.newaxis]*token_assignment/(total_tokens[:,np.newaxis])
            return -expected_winnings + np.max(np.sum(opponent_winnings,axis=0))
        

        optimal_solution = minimize(optimise_strategy, 10*np.ones(256)/256)
        sampled_tokens += 10*np.abs(optimal_solution.x)/np.sum(np.abs(optimal_solution.x))

    overall_expectation_data['our_tokens'] = 10*sampled_tokens/np.sum(sampled_tokens)

    # Ensure we place at least 1e-4 on each player
    overall_expectation_data['our_tokens'] = np.clip(overall_expectation_data['our_tokens'],1e-4,10)

    # Add in the lucky losers
    lucky_loser_dataframe = pd.DataFrame(
        {
            'player_name':lucky_losers,
            'our_tokens':[1e-7]*len(lucky_losers)
        }
    )

    overall_expectation_data = pd.concat([overall_expectation_data, lucky_loser_dataframe])
    
    # Final normalisation
    overall_expectation_data['our_tokens'] = overall_expectation_data['our_tokens']*10/np.sum(overall_expectation_data['our_tokens'])

    # Save
    overall_expectation_data.to_csv('final_token_selection.csv',index=False)



def get_ranking_based_expected_winnings(ranking:pd.Series) -> pd.Series:
    """
    Get ranking-based expected winnings
    """
    ranking_round_reached = np.floor(7 - np.log2(np.clip(ranking,1,127))).astype(int)
    prize_money = define_prize_money()
    return pd.Series({player:prize_money[round] for player, round in zip(ranking.index,ranking_round_reached )})



def load_demo_rankings() -> pd.Series:
    """
    Load a demo version of the draw for testing
    """
    results_df = pd.concat([pd.read_csv('src/main_package/data/2024_men.csv'),pd.read_csv('src/main_package/data/2024_women.csv')])
    wimbledon_first_round = results_df[results_df['Tournament'] == 'Wimbledon'].reset_index(drop=True)
    winner_ranking_map = wimbledon_first_round.set_index('Winner')['Rank'].to_dict()
    loser_ranking_map = wimbledon_first_round.set_index('Loser')['Rank'].to_dict()
    overall_ranking_map = {**winner_ranking_map, **loser_ranking_map}
    return pd.Series(overall_ranking_map)

def create_token_assignments(overall_expectation_data:pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create the token assignments from the random players
    """
    # Sample competitor Pareto parameters
    num_competitors = np.random.poisson(lam=20)
    alpha = np.random.random(size=num_competitors)*2 - 1

    # Sample expectations, allowing for bias from bookmaker's odds
    true_expectation = overall_expectation_data['mean_prize_money'].to_numpy()
    bookies_expectation = overall_expectation_data['ranking_based_money'].to_numpy()
    random_weights = np.random.random(size=num_competitors)
    player_expectations = true_expectation[:,np.newaxis]*random_weights[np.newaxis,:] + bookies_expectation[:,np.newaxis]*(1-random_weights[np.newaxis,:])

    # Sample player token assignments
    token_assignment = np.exp(alpha[np.newaxis,:]*player_expectations[:,np.newaxis])
    token_assignment = 10*token_assignment/(np.sum(token_assignment, axis=0)[np.newaxis,:])

    overall_expectation_data['tokens'] = np.sum(token_assignment, axis=1)
    return overall_expectation_data, token_assignment