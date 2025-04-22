import numpy as np
import pandas as pd
import os

class MeanModel:
    def __init__(self, input_steps=252, forecast_steps=7):
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.mean = None

    def fit(self, df, target_col):
        series = df[target_col].values
        if len(series) < self.input_steps:
            return False
        # Use only the last input_steps for mean calculation
        self.mean = np.mean(series[-self.input_steps:])
        return True

    def predict(self, input_series):
        # Predict the mean for the next forecast_steps
        if self.mean is None:
            raise ValueError("Model has not been fit yet.")
        return np.full(self.forecast_steps, self.mean)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, '..', '..', 'data', 'processed', 'intermediate')
    test_file = os.path.join(input_dir, 'AAPL_inter.csv')
    df = pd.read_csv(test_file)
    model = MeanModel()
    model.fit(df, target_col='close_pct_change')
    forecast = model.predict(df['close_pct_change'].values)
    print(forecast)