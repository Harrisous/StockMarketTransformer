from models.model_naive import MeanModel
from data_processing import read_index
import os
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, '..', 'data', 'processed', 'intermediate')
index_dir = os.path.join(script_dir, '..', 'data', 'raw')
ticker_list = read_index(os.path.join(index_dir, "0_ticker_index.json"))
results_dir = os.path.join(script_dir, "results")

def test():
    forecast_steps = 7
    input_steps = 252
    mse_accumulator = {}
    count_accumulator = {}
    counter = 0

    for ticker in ticker_list:
        counter += 1
        print(f"Processing {counter}/{len(ticker_list)}")
        file_path = os.path.join(input_dir, f"{ticker}_inter.csv")
        if not os.path.exists(file_path):
            print(file_path, "not exist")
            continue
        df = pd.read_csv(file_path)
        print(df.shape)
        columns = [
            "close_pct_change",
            "high_pct_change",
            "low_pct_change",
            "open_pct_change",
            "volume_pct_change"
        ]

        mse_accumulator_ticker = {col: 0.0 for col in columns}
        count_ticker = {col: 0 for col in columns}

        max_start = len(df) - input_steps - forecast_steps + 1
        # slice into train and backtesting set
        for start in range(max_start):
            train_idx = slice(start, start + input_steps)
            test_idx = slice(start + input_steps, start + input_steps + forecast_steps)
            if test_idx.stop > len(df):
                break

            for col in columns:
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                model = MeanModel(input_steps=input_steps, forecast_steps=forecast_steps)
                fit_success = model.fit(train_df, target_col=col)
                if not fit_success:
                    continue
                preds = model.predict(train_df[col].values)
                true = test_df[col].values
                mse = np.mean((preds - true) ** 2)
                mse_accumulator_ticker[col] += mse
                count_ticker[col] += 1

        for col in columns:
            if col not in mse_accumulator:
                mse_accumulator[col] = 0.0
                count_accumulator[col] = 0
            mse_accumulator[col] += mse_accumulator_ticker[col]
            count_accumulator[col] += count_ticker[col]

    avg_mse = {col: (mse_accumulator[col] / count_accumulator[col]) if count_accumulator[col] > 0 else None for col in mse_accumulator}

    # Save results
    result_path = os.path.join(results_dir, "mean_backtesting_mse.txt")
    with open(result_path, "w") as f:
        for col, mse in avg_mse.items():
            f.write(f"{col}: {mse}\n")

    # Return as a list of average MSEs in the order of columns
    mse_list = [avg_mse[col] for col in avg_mse]
    return mse_list

if __name__ == "__main__":
    test()