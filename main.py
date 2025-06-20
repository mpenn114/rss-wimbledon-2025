from src.main_package.model.fit_parameters import fit_parameters
from src.main_package.model.predict import predict_wimbledon_prize_money
from src.main_package.model.assign_tokens import assign_tokens
import pandas as pd
# fit_parameters(male_data=True)
# fit_parameters(male_data=False)
# male_prize_money = predict_wimbledon_prize_money(True)
# female_prize_money = predict_wimbledon_prize_money(False)

# combined_prize_money = pd.concat([male_prize_money,female_prize_money])
combined_prize_money = pd.read_csv('mock_combined_prize_money.csv')

assign_tokens(combined_prize_money)

