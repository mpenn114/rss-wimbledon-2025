### RSS Wimbledon 2025 Competition

This repository contains our entry to the Royal Statistical Society's 2025 Wimbledon competition (https://www-users.york.ac.uk/~bp787/pred_comp_2025.html)

### Data

Data was retrieved from http://www.tennis-data.co.uk/alldata.php on Saturday 28th June, 2025. Data from this website was converted to CSVs via https://cloudconvert.com/xlsx-to-csv for use in the code.

### Repository Setup

This repository is run using the environment manager `uv`, which can be installed via

Windows:

```bash
irm https://astral.sh/uv/install.ps1 | iex
```

Mac:

```bash
curl -Ls https://astral.sh/uv/install.sh | bash
```

Functions within this repository can then be run by using 

```bash
PYTHONPATH=. uv run path/to/functon.py
```

### Running the Code

The main pipeline function is found in `main.py`. This allows the user to create player strengths for the selected tournament, and perform forecasts of the prize money. Prize money forecasts are only possible where the draw is available - we include the Wimbledon 2025 draws in this repository. Draws should be added as a list of player names, in the order of the "tree" from that tournament.

There are some simple visualisations created in `src/main_package/notebooks/visualisation.ipynb`. These compare model performance to the ATP/WTA rankings and to the bookmakers' odds.

Note that to generate all the predictins needed for the longitudinal analysis in this notebook, you can run the following in the terminal (on Windows):

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\longitudinal_analysis.ps1
```