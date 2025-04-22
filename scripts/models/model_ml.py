import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os 

class LinearRegressionModel:
    def __init__(self, input_steps=252, forecast_steps=7):
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.model = LinearRegression()

    def create_supervised(self, series):
        X, y = [], []
        for i in range(len(series) - self.input_steps - self.forecast_steps + 1):
            X.append(series[i:i+self.input_steps])
            y.append(series[i+self.input_steps:i+self.input_steps+self.forecast_steps])
        return np.array(X), np.array(y)

    def fit(self, df, target_col):
        series = df[target_col].values
        # If exactly input_steps, fit a dummy model (predict last value repeated)
        if len(series) == self.input_steps:
            X = series.reshape(1, -1)
            y = np.zeros((1, self.forecast_steps))  # Dummy, as we can't fit real y
            self.model.fit(X, y)
            return True
        X, y = self.create_supervised(series)
        if X.size == 0 or y.size == 0:
            return False
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        self.model.fit(X, y)
        return True

    def predict(self, input_series):
        X_input = np.array(input_series[-self.input_steps:]).reshape(1, -1)
        return self.model.predict(X_input)[0]



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, '..', '..', 'data', 'processed', 'intermediate')

    test_file = os.path.join(input_dir, 'AAPL_inter.csv')
    df = pd.read_csv(test_file)
    model = LinearRegressionModel()
    model.fit(df, target_col='close_pct_change')
    forecast = model.predict(df['close_pct_change'].values)
    print(forecast)