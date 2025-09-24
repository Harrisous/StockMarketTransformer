import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.model_nn import TransformerForMultiStepPrediction
from data_processing import read_index
import numpy as np
import time 
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

# paths for saving
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', 'models')
checkpoint_dir = os.path.join(model_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

save_path = os.path.join(model_dir, "transformer_model.pth") # save for model
checkpoint_path = os.path.join(checkpoint_dir, "transformer_model_checkpoint_epoch_{epoch}.pth") # save for checkpoints during training

# tickers list
input_dir = os.path.join(script_dir, '..', 'data', 'raw')
index_path = os.path.join(input_dir,"0_ticker_index.json")
ticker_index_list = read_index(index_path)

# env_embedding path
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
env_embed_path = os.path.join(processed_dir, 'env_embeddings.csv')

# Hyperparameters
SEQ_LEN = 252
NUM_STEPS = 7
INPUT_DIM = 8  # stock dim (4) + env embedding dim (4)
OUTPUT_DIM = 4  # output dim
BATCH_SIZE = 256 - 64  # 256 - 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100 #****************important! Only training large epochs when time allows
LR = 1e-4
FILE_BATCH_SIZE = 100

class EnvSeqDataset(Dataset):
    def __init__(self, seq_len=252, num_steps=7, batch_size=FILE_BATCH_SIZE):
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ticker_ptr = 0
        self.samples = []
        self.env_df = pd.read_csv(env_embed_path)
        self.env_emb_dict = {str(row['date']): row.iloc[1:].values.astype('float32') for _, row in self.env_df.iterrows()}
        self.tickers = ticker_index_list
        self.load_next_batch()

    def load_next_batch(self):
        self.samples = []
        end_ptr = min(self.ticker_ptr + self.batch_size, len(self.tickers))
        for ticker in self.tickers[self.ticker_ptr:end_ptr]:
            csv_path = os.path.join(processed_dir, 'train', f'{ticker}_train.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            seq = []
            for _, row in df.iterrows():
                date = str(row['date'])
                stock_feat = row.iloc[1:5].values.astype('float32')
                if date in self.env_emb_dict:
                    env_feat = self.env_emb_dict[date]
                else:
                    raise ValueError(f"Date {date} not found in env_embeddings.csv")
                combined = np.concatenate([stock_feat, env_feat])
                seq.append(combined)
            seq = np.array(seq, dtype=np.float32)
            total_rows = len(seq)
            for start in range(0, total_rows - self.seq_len - self.num_steps + 1):
                x = torch.from_numpy(seq[start:start+self.seq_len]).clone().detach()
                y = torch.from_numpy(seq[start+self.seq_len:start+self.seq_len+self.num_steps, :4]).clone().detach()
                self.samples.append((x, y))
        self.ticker_ptr = end_ptr

        # print the current data loaded
        print("data loaded, total X, y pairs:", self.__len__())

    def next(self):
        if self.ticker_ptr >= len(self.tickers):
            return False  # No more data
        self.load_next_batch()
        return True
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y

# --- Training ---
def train(EPOCHS = EPOCHS, previou_model = None):
    model = TransformerForMultiStepPrediction(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        embed_dim=64, # 64, must be devided by num_heads
        num_heads=8, 
        num_layers=8, # 8
        hidden_dim=128, # 128,
        num_steps=NUM_STEPS,
        dropout=0.1,
        num_shared_experts=2,
        num_specialized_experts=6,
        seq_len=SEQ_LEN
    ).to(DEVICE)

    if previou_model is not None and os.path.exists(previou_model): # we can continue to train on previous model if specify the model
        print(f"Loading previous model from {previou_model}")
        model.load_state_dict(torch.load(previou_model, map_location=DEVICE))

    optimizer = optim.RMSprop(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()
    loss_log_path = os.path.join(model_dir, "nn_model_loss.log")
    with open(loss_log_path, "w") as log_file:
        log_file.write("epoch,loss,time\n")

    model.train()
    print("Model started training...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        dataset = EnvSeqDataset(seq_len=SEQ_LEN, num_steps=NUM_STEPS, batch_size=FILE_BATCH_SIZE) # adjust batch_size to optimize the usage of RAM
        epoch_loss = 0
        batch_count = 0
        while True:
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Batch {batch_count+1}")
            total_loss = 0
            for x, y in pbar:
                x = x.to(DEVICE, non_blocking=True) # x shape:[192, 252, 8]
                y = y.to(DEVICE) # y shape: [192, 7, 4]
                optimizer.zero_grad()
                out = model(x)
                if torch.isnan(out).any():
                    print("NaN in model output!")
                pred = out
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            epoch_loss += total_loss
            batch_count += 1
            scheduler.step()
            if not dataset.next():
                break  # No more batches
        
        avg_loss = epoch_loss / batch_count
        elapsed = time.time() - start_time
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}, time from start: {formatted_time}")
        with open(loss_log_path, "a") as log_file:
            log_file.write(f"{epoch+1},{avg_loss},{formatted_time}\n")
        

        # save checkpoint
        epoch_ckpt_path = checkpoint_path.format(epoch=epoch+1)
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"Checkpoint saved to: {epoch_ckpt_path}")

    # Save model
    print("train_finished")
    torch.save(model.state_dict(), save_path)
    print("model saved to:", save_path)

def test(batch_size):
    '''test batch size'''
    dataset = EnvSeqDataset(seq_len=SEQ_LEN, num_steps=NUM_STEPS, batch_size=batch_size)
    # x, y = dataset.__getitem__(0)
    # print("x shape", x.shape)
    # print("y shape", y.shape)
    while True:
        if not dataset.next():
            break

if __name__ == "__main__":
    # continue training from the third epoch from previous one
    # previous_model = os.path.join(checkpoint_dir, "transformer_model_checkpoint_epoch_b2.pth")
    train(EPOCHS= EPOCHS, previou_model=None) # use previous_model = save_path to load model and continue training
    # test(FILE_BATCH_SIZE)
    # data loaded, total X, y pairs: 658445
    # data loaded, total X, y pairs: 656830
    # data loaded, total X, y pairs: 640403
    # data loaded, total X, y pairs: 677345
    # data loaded, total X, y pairs: 660781
    # data loaded, total X, y pairs: 2198
    # total training datapint: 3,296,002
