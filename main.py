from src.main_package.model.train import train_model

overall_rmse = train_model()
print(f"Overall RMSE score: {overall_rmse}")
