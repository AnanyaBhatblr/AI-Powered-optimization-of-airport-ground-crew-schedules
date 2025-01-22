# scripts/arima_forecasting.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

def forecast_crew_demand(file_path):
    tasks = pd.read_csv(file_path)
    crew_demand = tasks.groupby('Start Time').size().resample('D').sum()

    model = ARIMA(crew_demand, order=(2, 1, 2))
    model_fit = model.fit()

    with open('../models/arima_model.pkl', 'wb') as f:
        pickle.dump(model_fit, f)

    print("ARIMA model trained and saved!")

if __name__ == "__main__":
    forecast_crew_demand("../data/processed_tasks.csv")
