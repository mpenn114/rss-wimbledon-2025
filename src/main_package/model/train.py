import pandas as pd


def train_model():
    """
    Create player feature list
    """
    # Training set is everything up to Wimbledon on each year
    # Test set is predicting Wimbledon
    # We create player ability at Wimbledon = prod(exp(alpha_i * variable_i))
    # P(win match) = 1/(1 + 10^{Ability Difference})
    # Our objective is sum of squared differences in Wimbledon prize money
