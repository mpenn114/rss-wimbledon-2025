# Configuration
$scriptName = "alternative_models.py"
$models = @('bradley_terry_model', 'elo_model', 'random_forest_model')
$tournaments = @('Australian Open', 'French Open', 'Wimbledon', 'US Open')
$years = @(2024, 2025)

foreach ($model in $models) {
    foreach ($tournament in $tournaments) {
        foreach ($year in $years) {
            
            # Exclusion logic: Skip US Open for the year 2025
            if ($tournament -eq "US Open" -and $year -eq 2025) {
                Write-Host "Skipping $tournament for $year as requested." -ForegroundColor Yellow
                continue
            }

            # 1. Run Male Data (Default)
            Write-Host "Executing: $model | $tournament | $year | Male" -ForegroundColor Cyan
            uv run $scriptName --model $model --tournament "$tournament" --year $year

            # 2. Run Female Data (Using the --female flag)
            Write-Host "Executing: $model | $tournament | $year | Female" -ForegroundColor Magenta
            uv run $scriptName --model $model --tournament "$tournament" --year $year --female
        }
    }
}

Write-Host "Batch processing complete!" -ForegroundColor Green