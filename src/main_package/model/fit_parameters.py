from src.main_package.model.train import train_model
import pandas as pd
from typing import List
def fit_parameters(male_data:bool = True):
    """
    Fit the model parameters through a simple grid-based search

    Args:
        male_data (bool): Whether or not we are looking at male data
    """
    temporal_decay_grid = [0.05,0.1,0.2]
    grass_weight_grid = [2.0,4.0,6.0,8.0]
    parameter_evals:List[pd.Series] = []
    for temporal in temporal_decay_grid:
        for grass in grass_weight_grid:
            overall_rmse,_ = train_model(temporal, grass, male_data=male_data)
            parameter_evals.append(pd.Series(
                {
                    'temporal_decay':temporal,
                    'grass_weight':grass,
                    'overall_rmse':overall_rmse
                }
            ))
            print(parameter_evals[-1])
    overall_parameters = pd.concat(parameter_evals,axis=1).T
    suffix = 'male' if male_data else 'female'
    overall_parameters.to_csv(f"parameter_evals_{suffix}.csv",index=False)
