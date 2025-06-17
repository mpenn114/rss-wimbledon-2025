import numpy as np

def define_prize_money() -> np.ndarray:
    """
    Define the prize money for reaching each round

    Returns:
        np.ndarray: The prize money by round
    """
    return np.array([
        66_000,
        99_000,
        152_000,
        240_000,
        400_000,
        775_000,
        1_520_000,
        3_000_000
    ])