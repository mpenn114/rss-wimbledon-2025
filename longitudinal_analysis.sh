# Note: We weight matches on the target surface 20 times more for male players and 50 times more for female players, based on our parameter fitting
uv run main.py --tournament "Australian Open" --tournament-year 2024 --clay-weight-male 0.05 --clay-weight-female 0.02 --grass-weight-male 0.05 --grass-weight-female 0.02
uv run main.py --tournament "French Open" --tournament-year 2024 --clay-weight-male 20.0 --clay-weight-female 50.0 --grass-weight-male 1.0 --grass-weight-female 1.0
uv run main.py --tournament "Wimbledon" --tournament-year 2024 --clay-weight-male 1.0 --clay-weight-female 1.0 --grass-weight-male 20.0 --grass-weight-female 50.0
uv run main.py --tournament "US Open" --tournament-year 2024 --clay-weight-male 0.05 --clay-weight-female 0.02 --grass-weight-male 0.05 --grass-weight-female 0.02
uv run main.py --tournament "Australian Open" --tournament-year 2025 --clay-weight-male 0.05 --clay-weight-female 0.02 --grass-weight-male 0.05 --grass-weight-female 0.02
uv run main.py --tournament "French Open" --tournament-year 2025 --clay-weight-male 20.0 --clay-weight-female 50.0 --grass-weight-male 1.0 --grass-weight-female 1.0
uv run main.py --tournament "Wimbledon" --tournament-year 2025 --clay-weight-male 1.0 --clay-weight-female 1.0 --grass-weight-male 20.0 --grass-weight-female 50.0
