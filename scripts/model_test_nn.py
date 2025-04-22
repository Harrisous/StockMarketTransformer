import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.model_nn import TransformerForMultiStepPrediction
from models.autoencoder import Autoencoder
from data_processing import read_index
import joblib

# Hyperparameters (should match training)
SEQ_LEN = 252
NUM_STEPS = 7
INPUT_DIM = 8
OUTPUT_DIM = 4
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset for test data
class TestSeqDataset(Dataset):
    def __init__(self, csv_path, env_emb_dict):
        # read from embedding
        self.samples = []
        self.first_date = None

        df = pd.read_csv(csv_path)
        seq = []
        for _, row in df.iterrows():
            date = str(row['date'])
            stock_feat = row.iloc[1:5].values.astype('float32')
            env_feat = env_emb_dict.get(date) # concatonate with lastest env embedding

            if env_feat is None:
                print("env_fest is None")
                continue

            # save the first date for matching with ground truth
            if self.first_date is None:
                    self.first_date = date

            combined = np.concatenate([stock_feat, env_feat])
            seq.append(combined)

            
        seq = np.array(seq, dtype=np.float32)
        total_rows = len(seq)
        for start in range(0, total_rows - SEQ_LEN - NUM_STEPS + 1):
            x = torch.from_numpy(seq[start:start+SEQ_LEN]).clone().detach()
            self.samples.append(x)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_first_date(self):
        return self.first_date
    
def test():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # model param file
    model_dir = os.path.join(script_dir, '..', 'models')
    selected_ckpt = os.path.join(model_dir, 'selected', 'transformer_model_checkpoint_epoch_b2.pth')

    # autoencoder and scaler
    autoencoder_path = os.path.join(model_dir, 'ticker_autoencoder.pth')
    scaler_path = os.path.join(model_dir, 'ticker_scaler.save')

    # test data (embedded)
    test_data_dir = os.path.join(script_dir, '..', 'data', 'processed', 'test') # 1/2 X_test
    env_embed_path = os.path.join(script_dir, '..', 'data', 'processed', 'env_embeddings.csv') # another 1/2 X_test
    intermediate_dir = os.path.join(script_dir, '..', 'data', 'processed', 'intermediate') # truth
    results_dir = os.path.join(script_dir, 'results')

    # ticker_index
    ticker_index_path = os.path.join(script_dir, '..', 'data', 'raw', '0_ticker_index.json')

    # Load scaler and autoencoder
    scaler = joblib.load(scaler_path)
    autoencoder = Autoencoder(input_dim=6, latent_dim=4, size=4).to(DEVICE) # size = 4 for ticker data encoder 
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=DEVICE))
    autoencoder.eval()

    # Load transformer model
    model = TransformerForMultiStepPrediction(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        embed_dim=64,
        num_heads=8,
        num_layers=8,
        hidden_dim=128,
        num_steps=NUM_STEPS,
        dropout=0.1,
        num_shared_experts=2,
        num_specialized_experts=6,
        seq_len=SEQ_LEN
    ).to(DEVICE)
    model.load_state_dict(torch.load(selected_ckpt, map_location=DEVICE))
    model.eval() # turn to eval mode to make predictions

    # Load env embeddings
    env_df = pd.read_csv(env_embed_path)
    env_emb_dict = {str(row['date']): row.iloc[1:].values.astype('float32') for _, row in env_df.iterrows()}

    # Get tickers
    tickers = read_index(ticker_index_path)
    columns = ["close_pct_change", "high_pct_change", "low_pct_change", "open_pct_change", "volume_pct_change"]
    mse_accumulator = {col: 0.0 for col in columns}
    count_accumulator = {col: 0 for col in columns}

    for ticker in tqdm(tickers, desc="Testing NN"):
        test_csv = os.path.join(test_data_dir, f"{ticker}_test.csv")
        inter_csv = os.path.join(intermediate_dir, f"{ticker}_inter.csv")
        if not os.path.exists(test_csv) or not os.path.exists(inter_csv):
            print(test_csv, "path not exist")
            continue

        # Prepare dataset and dataloader
        dataset = TestSeqDataset(test_csv, env_emb_dict)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        preds_all = []
        for x in dataloader:
            x = x.to(DEVICE)
            with torch.no_grad():
                out = model(x)  # [B, NUM_STEPS, OUTPUT_DIM]
                out = out.view(-1, OUTPUT_DIM)
                decoded = autoencoder.decoder(out)  # [B*NUM_STEPS, OUTPUT_DIM]
                decoded = decoded.cpu().numpy()
                decoded = decoded[:,1:] # remove the predicted ticker index

                # Inverse transform scaler
                decoded = scaler.inverse_transform(decoded) # shape: (7,5)
                preds_all.append(decoded)

        # preds_all size: (N, 7, 5)
        # Get ground truth
        df_inter = pd.read_csv(inter_csv) # only the last 5 columns are matching
        first_date = dataset.get_first_date()  
        first_idx = df_inter.index[df_inter['date'] == first_date].tolist()
        # remove previous date to align size with X_test
        if first_idx:
            df_inter = df_inter.iloc[first_idx[0]:].reset_index(drop=True)
        else:
            print(f"Warning: first_date {first_date} not found in ground truth for {ticker}")
        gt = []
        total_rows = len(df_inter)
        # read ground truth
        for start in range(0, total_rows - SEQ_LEN - NUM_STEPS + 1):
            y = df_inter.iloc[start+SEQ_LEN:start+SEQ_LEN+NUM_STEPS][columns].values
            gt.append(y)
        # Skip this ticker if no valid ground truth or predictions eg. ABNB
        if len(gt) == 0 or len(preds_all) == 0:
            print(f"Skipping {ticker}: no valid ground truth or predictions.")
            continue
        # gt size & preds_all size: (N,7,5) -> flat
        gt_flat = np.vstack(gt)
        preds_flat = np.vstack(preds_all)

        # Compute MSE for each feature
        for i, col in enumerate(columns):
            mse = np.mean((preds_flat[:, i] - gt_flat[:, i]) ** 2)
            mse_accumulator[col] += mse * len(gt_flat)
            count_accumulator[col] += len(gt_flat)

    # Average MSE
    avg_mse = {col: (mse_accumulator[col] / count_accumulator[col]) if count_accumulator[col] > 0 else None for col in columns}

    # Log results
    result_path = os.path.join(results_dir, "nn_test_mse.txt")
    with open(result_path, "w") as f:
        for col, mse in avg_mse.items():
            f.write(f"{col}: {mse}\n")

if __name__ == "__main__":
    test()