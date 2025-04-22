'''
an app page containing 3 different methods. The title is "Financial Time Series Forecasting", 
there should be a drop-down search bar for the user to enter the stock ticker,
once get the ticker, the script should first get the current date, then fetch the past 252 (not including today) rows using yfinance
plot the last 252 rows of data into candel bar using open, high, low, close, the y axis should be self-adjusting according to the max value
and also plot the volumes in a sparate graph.
'''
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objs as go
import os 
import json
import torch
from scripts.models.model_naive import MeanModel
from scripts.models.model_ml import LinearRegressionModel
from scripts.models.model_nn import TransformerForMultiStepPrediction
from scripts.models.autoencoder import Autoencoder
import joblib
import numpy as np

# functions
def read_index(index_path):
    '''read the index json file and return a list containing lists'''
    with open(index_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data

# load available stock tickers
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, 'data', 'raw')
index_path = os.path.join(raw_dir,"0_ticker_index.json")
tickers = read_index(index_path)
ticker_index_list = sorted(tickers)

# model paths
models_dir = os.path.join(script_dir, "app_models")

env_embeddings_path =  os.path.join(models_dir, 'env_embeddings.csv')
selected_model_path = os.path.join(models_dir, 'transformer_model_checkpoint_epoch_b2.pth')
ticker_scaler_path = os.path.join(models_dir, 'ticker_scaler.save')
ticker_autoencoder_path = os.path.join(models_dir, 'ticker_autoencoder.pth')
# not implemented for the time being
# env_scaler_path = os.path.join(models_dir, 'ticker_scaler.save') 
# env_autoencoder_path = os.path.join(models_dir, 'env_autoencoder.pth') 


# Hyperparameters (should match training)
SEQ_LEN = 252
NUM_STEPS = 7
INPUT_DIM = 8
OUTPUT_DIM = 4
BATCH_SIZE = 1
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

@st.cache_resource
def load_transformer_resources():
    # load model
    transformer = TransformerForMultiStepPrediction(
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
    transformer.load_state_dict(torch.load(selected_model_path, map_location=DEVICE))

    # load scaler related scaler and autoencoder
    ticker_scaler = joblib.load(ticker_scaler_path)
    ticker_autoencoder = Autoencoder(input_dim=6, latent_dim=4, size=4).to(DEVICE)
    ticker_autoencoder.load_state_dict(torch.load(ticker_autoencoder_path, map_location=DEVICE))
    ticker_autoencoder.eval()

    # env scaler and related scaler and autoencoder will be using exsiting embeddings
    # env_scaler = joblib.load(env_scaler_path)
    # env_autoencoder = Autoencoder(input_dim=?, latent_dim=4, size=64).to(DEVICE)

    return transformer, ticker_scaler, ticker_autoencoder # ,env_scaler, env_autoencoder

@st.cache_resource
def get_env_embedding(): 
    return pd.read_csv(env_embeddings_path)

# page start
st.set_page_config(page_title="Financial Time Series Forecasting", layout="wide")
st.title("Financial Time Series Forecasting")
st.write("contact: hl535@duke.edu")

# two selection boxed: ticker selection and mode selection
col1, col2 = st.columns([3, 2])
with col1:
    selected_ticker = st.selectbox("Select or enter a stock ticker:", ticker_index_list, index=0)
with col2:
    method = st.selectbox("Select forecasting method:", ["Plain", "Mean", "Linear", "Transformer"], index=0)

# plot the graph accordingly
if selected_ticker:
    today = datetime.datetime.now().date()
    # Fetch 252 trading days and generating pct change for estimation
    data = yf.download(selected_ticker, period="2y")  # Fetch more to ensure 252 rows
    
    percentage_change = data.pct_change().dropna()
    
    data = data.tail(252)
    percentage_change = percentage_change.tail(252)

    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    st.subheader(f"Last 252 Trading Days for {selected_ticker}")

    # Candlestick chart for OHLC
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    forecast_steps = 7
    input_steps = 252

    # Volume bar chart
    st.subheader("Volume")
    fig_vol = go.Figure(data=[go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color='blue'
    )])
    
    # handle transformer
    if method == "Transformer":
        # Load saved models
        transformer, ticker_scaler, ticker_autoencoder = load_transformer_resources()

        # scale and add idx 
        ticker_encoding = percentage_change
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = ticker_scaler.transform(ticker_encoding[columns]) # (252,5)
        idx = ticker_index_list.index(selected_ticker)
        idx_column = np.full((features.shape[0], 1), idx)  # shape (252,1)
        features_with_idx = np.concatenate([idx_column, features], axis=1)  # shape (252,6) -> add idx
        features_tensor = torch.tensor(features_with_idx, dtype=torch.float32).to(DEVICE) # (252, 6)
        # embed autoencoder
        with torch.no_grad():
            ticker_encoded = ticker_autoencoder.encoder(features_tensor).cpu().numpy()
        ticker_encoded = torch.tensor(ticker_encoded, dtype=torch.float32).to(DEVICE)    

        # get environment embedding 
        env_emb = get_env_embedding()
        env_emb['date'] = pd.to_datetime(env_emb['date'])
        dates_df = pd.DataFrame({'date': percentage_change.index})
        merged = dates_df.merge(env_emb, on='date', how='left')
        
        embedding_cols = [col for col in env_emb.columns if col != 'Date'] # fill missing embeddings with the last row of env_emb
        for col in embedding_cols:
            merged[col] = merged[col].fillna(env_emb[col].iloc[-1])
        
        env_tensor = torch.tensor(merged.iloc[:, -4:].values, dtype=torch.float32).to(DEVICE)
        # ticker_encoded: (252,4) 
        # env_tensor: (252,4)

        # concatonate
        model_input_tensor = torch.cat([ticker_encoded, env_tensor], dim=1) # (252, 8)
        model_input_tensor = model_input_tensor.unsqueeze(0) 
        # predict 
        model_input_tensor = model_input_tensor.to(DEVICE)
        with torch.no_grad(): 
            out = transformer(model_input_tensor)  # [B, NUM_STEPS, OUTPUT_DIM]
            out = out.view(-1, OUTPUT_DIM)
            decoded = ticker_autoencoder.decoder(out)  # [B*NUM_STEPS, OUTPUT_DIM]
            decoded = decoded.cpu().numpy()
            decoded = decoded[:,1:] # remove the predicted ticker index

            # Inverse transform scaler
            decoded = ticker_scaler.inverse_transform(decoded) # shape: (7,5)
            
            # 1. Create DataFrame for predicted percentage changes
            pred_dates = pd.bdate_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
            pred_pct_df = pd.DataFrame(decoded, columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=pred_dates)

            # 2. Convert percentage changes to actual values
            last_row = data.iloc[-1]
            pred_df = pd.DataFrame(columns=data.columns, index=pred_pct_df.index)
            prev = last_row.copy()
            for i, d in enumerate(pred_pct_df.index):
                pred = prev * (1 + pred_pct_df.iloc[i])
                pred_df.loc[d] = pred
                prev = pred

            # 3. Add predicted candles to the plot
            for i in range(forecast_steps):
                color = 'black' if pred_df['Close'].iloc[i] >= pred_df['Open'].iloc[i] else 'gray'
                fig_candle.add_trace(go.Candlestick(
                    x=[pred_df.index[i]],
                    open=[pred_df['Open'].iloc[i]],
                    high=[pred_df['High'].iloc[i]],
                    low=[pred_df['Low'].iloc[i]],
                    close=[pred_df['Close'].iloc[i]],
                    increasing_line_color='white',
                    decreasing_line_color='gray',
                    showlegend=False
                ))

            # 4. Add predicted volumes to the volume plot
            pred_volumes = pred_df['Volume']
            pred_colors = ['black' if pred_df['Close'].iloc[i] >= pred_df['Open'].iloc[i] else 'gray' for i in range(forecast_steps)]
            fig_vol.add_trace(go.Bar(
                x=pred_df.index,
                y=pred_volumes,
                marker_color=pred_colors,
                opacity=0.7,
                name='Predicted Volume'
            ))


    if method == "Mean" or method == "Linear":
        pred_dates = pd.bdate_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
        pred_pct_df = pd.DataFrame(index=pred_dates)
        model_class = MeanModel if method == "Mean" else LinearRegressionModel
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            model = model_class(input_steps=input_steps, forecast_steps=forecast_steps)
            fit_success = model.fit(percentage_change, target_col=col)
            if not fit_success:
                print("not success")
                pred_pct_df[col] = [0]*forecast_steps
            else:
                preds = model.predict(percentage_change[col].values)
                pred_pct_df[col] = preds

        # Transform percentage change predictions to price/volume
        last_row = data.iloc[-1]
        pred_df = pd.DataFrame(columns=data.columns, index=pred_pct_df.index)
        prev = last_row.copy()
        for i, d in enumerate(pred_pct_df.index):
            pred = prev * (1 + pred_pct_df.iloc[i])
            pred_df.loc[d] = pred
            prev = pred

        # Add predicted candles to the plot
        for i in range(forecast_steps):
            color = 'black' if pred_df['Close'].iloc[i] >= pred_df['Open'].iloc[i] else 'gray'
            fig_candle.add_trace(go.Candlestick(
                x=[pred_df.index[i]],
                open=[pred_df['Open'].iloc[i]],
                high=[pred_df['High'].iloc[i]],
                low=[pred_df['Low'].iloc[i]],
                close=[pred_df['Close'].iloc[i]],
                increasing_line_color='white',
                decreasing_line_color='gray',
                showlegend=False
            ))

    fig_candle.update_layout(
        yaxis_title='Price',
        xaxis_title='Date',
        yaxis=dict(autorange=True),
        height=500,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    

    if method == "Mean" or method == "Linear":
        pred_volumes = pred_df['Volume']
        pred_colors = ['black' if pred_df['Close'].iloc[i] >= pred_df['Open'].iloc[i] else 'gray' for i in range(forecast_steps)]
        fig_vol.add_trace(go.Bar(
            x=pred_df.index,
            y=pred_volumes,
            marker_color=pred_colors,
            opacity=0.7,
            name='Predicted Volume'
        ))

    fig_vol.update_layout(
        yaxis_title='Volume',
        xaxis_title='Date',
        height=200,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_vol, use_container_width=True)