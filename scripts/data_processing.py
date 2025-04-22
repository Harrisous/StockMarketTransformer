import json
import os
import pandas as pd
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder import Autoencoder
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
import joblib
from torch.optim.lr_scheduler import StepLR

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, '..', 'data', 'raw')
output_dir = os.path.join(script_dir, '..', 'data', 'processed')
model_dir = os.path.join(script_dir, '..', 'models')

def read_index(index_path):
    '''read the index json file and return a list containing lists'''
    with open(index_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data

def data_processing_nn():
    # step1: process each stock to be the increase precentage
    index_path = os.path.join(input_dir,"0_ticker_index.json")
    ticker_index_list = read_index(index_path)
    combined_data = pd.DataFrame() 

    for i in range(len(ticker_index_list)):
        # read index
        ticker_file = os.path.join(input_dir, f"{ticker_index_list[i]}_raw.csv")
        data = pd.read_csv(ticker_file)
        # feature creation
        cols_to_pct_change = ['close', 'high', 'low', 'open', 'volume']
        for col in cols_to_pct_change:
            if col in data.columns:
                data[f'{col}_pct_change'] = data[col].pct_change()
        data.fillna(0, inplace=True) # fix NAN data
        data = data.replace([np.inf, -np.inf], 0) # fix inf data
        data["ticker_idx"] = i # add stock id
        data = data.iloc[1:] # first row should be removed 
        data = data[["ticker_idx", "date", "close_pct_change",  "high_pct_change", "low_pct_change", "open_pct_change", "volume_pct_change"]]
        # temp save
        export_dir = os.path.join(output_dir, "intermediate", f"{ticker_index_list[i]}_inter.csv")
        data.to_csv(export_dir)
        # add combined_data for autoencoder training
        data = data[["ticker_idx", "close_pct_change",  "high_pct_change", "low_pct_change", "open_pct_change", "volume_pct_change"]]
        if data.empty:
            print(ticker_index_list[i])
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    # normalization
    scaler = StandardScaler()
    combined_data[["close_pct_change", "high_pct_change", "low_pct_change", "open_pct_change", "volume_pct_change"]] = \
    scaler.fit_transform(combined_data[["close_pct_change", "high_pct_change", "low_pct_change", "open_pct_change", "volume_pct_change"]])
    scaler_path = os.path.join(model_dir, "ticker_scaler.save")
    joblib.dump(scaler, scaler_path)
    # remove extreme values
    combined_data = combined_data.clip(lower=-5, upper=5)
    
    # step2: train autoencoder based on stock index, close, high, low, open, volume
    model = train_autoencoder(combined_data, LEARNING_RATE = 1e-3, INPUT_DIM = combined_data.shape[1], NUM_EPOCHS = 50, BATCH_SIZE = 1024*1024, size = 4, encoder_type = "ticker")

    # step3: apply autoencoder to ticker data and save to train and test folder
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LATENT_DIM = 4

    for i in range(len(ticker_index_list)):
        export_dir = os.path.join(output_dir, "intermediate", f"{ticker_index_list[i]}_inter.csv")
        df = pd.read_csv(export_dir)
        features = df[["ticker_idx", "close_pct_change", "high_pct_change", "low_pct_change", "open_pct_change", "volume_pct_change"]].values
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            encoded = model.encoder(features_tensor).cpu().numpy()
        # Create encoded DataFrame, keep date and ticker_idx for reference
        encoded_df = pd.DataFrame(encoded, columns=[f"enc_{j}" for j in range(LATENT_DIM)])
        encoded_df["ticker_idx"] = df["ticker_idx"].values
        encoded_df["date"] = df["date"]
        # Reorder columns: date, ticker_idx, enc_0, enc_1, ...
        cols = ["date", "ticker_idx"] + [f"enc_{j}" for j in range(LATENT_DIM)]
        encoded_df = encoded_df[cols]

        # Split 8:2 train:test (no shuffle)
        n = len(encoded_df)
        n_train = int(n * 0.8)
        train_df = encoded_df.iloc[:n_train]
        test_df = encoded_df.iloc[n_train:]

        # Save
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        train_df.to_csv(os.path.join(train_dir, f"{ticker_index_list[i]}_train.csv"), index=False)
        test_df.to_csv(os.path.join(test_dir, f"{ticker_index_list[i]}_test.csv"), index=False)


def env_data_processing_nn():
    '''process environmental indices'''
    # step 1: load and concatonate environment vectors
    market_index_path = os.path.join(input_dir, "0_market_index.json")
    macro_index_path = os.path.join(input_dir, "0_macro_index.json")

    # Read index files to get all environmental indicator names
    with open(market_index_path, "r") as f:
        market_indices = json.load(f)
    with open(macro_index_path, "r") as f:
        macro_indices = json.load(f)
    all_indices = market_indices + macro_indices  # List of indicator names

    # Tickers with OHLCV(open, high, low, close, volumn) columns
    ohlcv_tickers = {
        'S&P500', 'DJI', 'Nasdaq', 'gold_futures', 'crude_oil_futures',
        '13_Week_Treasury_Bill', '26_Week_Treasury_Bill', '52_Week_Treasury_Bill',
        '30_Year_Treasury_Bond', 'US_Dollar_Index'
    }

    data_dict = {}
    date_set = set()

    for idx in all_indices:
        file_path = os.path.join(input_dir, f"{idx}_raw.csv")
        df = pd.read_csv(file_path)
        # Remove unnamed columns (like the index column)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # Try to find the date column robustly
        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
        if date_col is None:
            raise ValueError(f"No date column found in {file_path}. Columns: {df.columns.tolist()}")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        if idx in ohlcv_tickers:
            if idx in ohlcv_tickers:
            # Add columns with ticker prefix for OHLCV
                for col in ['close', 'high', 'low', 'open', 'volume']:
                    if col in df.columns:
                        df[f"{idx}_{col}"] = df[col]
                keep_cols = [date_col] + [f"{idx}_{col}" for col in ['close', 'high', 'low', 'open', 'volume'] if f"{idx}_{col}" in df.columns]
                df = df[keep_cols]
        else:
            df = df[[date_col, df.columns[-1]]]
            df.columns = ["date", idx]
        data_dict[idx] = df
        date_set.update(df["date"].tolist())

    all_dates = sorted(list(date_set))
    start_date = all_dates[0]
    end_date = all_dates[-1]
    daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    env_df = pd.DataFrame({"date": daily_dates})

    # Merge each indicator, forward-fill to fill missing values for daily frequency
    for idx, df in data_dict.items():
        env_df = env_df.merge(df, on="date", how="left")
        # Forward fill all new columns except 'date'
        new_cols = [col for col in df.columns if col != "date"]
        for col in new_cols:
            env_df[col] = env_df[col].ffill()

    # Do pct_change for all columns except real_GDP and Unemployment_Rate
    exclude_cols = {"real_GDP", "Unemployment_Rate"}
    for col in env_df.columns:
        if col not in exclude_cols and col != "date":
            env_df[col] = env_df[col].pct_change()
    # Fix NaN and inf
    env_df = env_df.replace([np.inf, -np.inf], 0)
    env_df = env_df.fillna(0)

    # Save to processed
    os.makedirs(output_dir, exist_ok=True)
    env_df.to_csv(os.path.join(output_dir, "env_processed.csv"), index=False)
    print(f"Environmental data processed and saved to {os.path.join(output_dir, 'env_processed.csv')}")

    # step2: train autoencoder for environment variable
    env_features = env_df.drop(columns=["date"])
    # normalization
    scaler = StandardScaler()
    env_features_scaled = scaler.fit_transform(env_features)
    scaler_path = os.path.join(model_dir, "env_scaler.save")
    joblib.dump(scaler, scaler_path)
    env_features_scaled = np.clip(env_features_scaled, -5, 5)
    env_features_scaled_df = pd.DataFrame(env_features_scaled, columns=env_features.columns)

    # train model
    model = train_autoencoder(env_features_scaled_df, LEARNING_RATE = 1e-4, INPUT_DIM=env_features_scaled_df.shape[1], NUM_EPOCHS=50, BATCH_SIZE=1024*1024, size=64, encoder_type="env")

    # get and save the df with only dimension embeddings
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LATENT_DIM = 4

    features_tensor = torch.tensor(env_features_scaled_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        encoded = model.encoder(features_tensor).cpu().numpy()

    encoded_df = pd.DataFrame(encoded, columns=[f"env_enc_{i}" for i in range(LATENT_DIM)])
    encoded_df["date"] = env_df["date"].values
    cols = ["date"] + [f"env_enc_{i}" for i in range(LATENT_DIM)]
    encoded_df = encoded_df[cols]

    encoded_df.to_csv(os.path.join(output_dir, "env_embeddings.csv"), index=False)
    print(f"Environmental embeddings saved to {os.path.join(output_dir, 'env_embeddings.csv')}")


def train_autoencoder(data:pd.DataFrame, LEARNING_RATE, INPUT_DIM:int, NUM_EPOCHS:int, BATCH_SIZE:int, size, encoder_type:str):
    '''
    param: INPUT_DIM # Number of features. For ticker: (ticker_idx, close_pct, high_pct, low_pct, open_pct, volume_pct)
    '''
    LATENT_DIM = 4 # Example latent dimension (bottleneck size)

    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, size=size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # prepare log
    loss_log_path = os.path.join(model_dir, f"{encoder_type}_autoencoder_loss.log")
    with open(loss_log_path, "w") as log_file:
        log_file.write("epoch,loss\n")

    print("Starting training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_data in dataloader:
            inputs = batch_data[0].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}, Time elapsed: {elapsed:.2f}s")
        # Append loss to log file
        with open(loss_log_path, "a") as log_file:
            log_file.write(f"{epoch+1},{avg_loss}\n")
        scheduler.step()

    print("Training finished.")
    model_save_path = os.path.join(model_dir, f"{encoder_type}_autoencoder.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

if __name__ == "__main__":
    # nn
    # step1 process ticker_data
    data_processing_nn()
    # step2 process env data
    env_data_processing_nn()