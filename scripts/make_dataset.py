'''
This file is used to download the whole database to local machine. Alpha Vantage free tier is only available for using 20 year data so is passed.
Tip: this process may take around 2-3 min
'''
import pandas as pd
import os
import yfinance as yf
from pandas_datareader import data as pdr
import datetime
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..', 'data', 'raw')



def de_multi_indexing_yf(stock_data):
    '''Function to solve multi-indexing issue from yf'''
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    if stock_data.index.name == 'Date':
        stock_data = stock_data.reset_index()
    stock_data.columns = [col.lower() for col in stock_data.columns]

    return stock_data


def save_index(index_list:list, save_name:str):
    '''this function is used to save the index list as the save_name'''
    output_file = os.path.join(output_dir, f'{save_name}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f: # override
        json.dump(index_list, f)
    print(f"{save_name} saved successfully")

def fetch_stock():
    '''function to auto fetch stock indices'''
    # get all stock indices from 
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]  # Read the first table from the webpage
    sp500_tickers = sp500_table['Symbol'].tolist()

    counter = 0
    sp500_tickers_valid = []
    for symbol in sp500_tickers:
        print(f"Processing symbol: {symbol}")
        try:
            # fetch stock data
            stock_data = yf.download(symbol, period="max")
            if not stock_data.empty: # exclude all empty datasets
                # solve the multi-indexing issue
                stock_data = de_multi_indexing_yf(stock_data)

                # save data to local
                output_csv_file = os.path.join(output_dir, f"{symbol}_raw.csv")
                stock_data.to_csv(output_csv_file)

                counter += 1 # only count when valid
                sp500_tickers_valid.append(symbol)
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")

    # save the index 
    save_index(sp500_tickers_valid, "0_ticker_index")
    print("stock loaded, total valid stock count:", counter) 



def fetch_market_indices():
    index_map_yf = {
        # three most important stock indices
        'S&P500': '^GSPC', 
        'DJI': '^DJI',
        'Nasdaq': '^IXIC',
        # oil and gold
        'gold_futures': 'GC=F',  # gold futures
        'crude_oil_futures': 'CL=F',  # Crude Oil Futures
        # USD related
        '13_Week_Treasury_Bill': '^IRX',  # 13 Week Treasury Bill
        '26_Week_Treasury_Bill': '^FVX',  # 26 Week Treasury Bill (5-Year Yield)
        '52_Week_Treasury_Bill': '^TNX',  # 52 Week Treasury Bill (10-Year Yield)
        '30_Year_Treasury_Bond': '^TYX',  # 30-Year Treasury Bond
        'US_Dollar_Index': 'DX-Y.NYB'  # US Dollar Index
    }
    index_list = []
    for index_name, ticker in index_map_yf.items():
        try:
            # fetch data
            data = yf.download(ticker, period="max")  # Fetch all available data
            print(f"Data for {index_name} fetched successfully.")
            # multi-indexing issue
            data = de_multi_indexing_yf(data)

            # save data
            data_dir = os.path.join(output_dir, f"{index_name}_raw.csv")
            data.to_csv(data_dir)

            index_list.append(index_name)

        except Exception as e:
            print(f"Failed to fetch data for {index_name}: {e}")
    
    # save the index
    save_index(index_list,"0_market_index") 



def fetch_macro_indices():
    series_ids  = {
        # macro econ
        'Real_GDP': 'A191RL1Q225SBEA',  # Real Gross Domestic Product, Quarterly, SAAR; return: percentage change from preceding Quater
        'CPI': 'CPIAUCSL', # Consumer Price Index for All Urban Consumers: All Items, Monthly, SA
        'Unemployment_Rate': 'UNRATE' # Civilian Unemployment Rate, Monthly, SA
    }

    start = datetime.datetime(1900, 1, 1)
    end = datetime.datetime.today()
    valid_list = []
    for index_name, id in series_ids.items():
        try: 
            # get data
            data = pdr.DataReader(id, 'fred', start, end)
            print(f"Data for {index_name} fetched successfully.")

            # formatting
            if data.index.name == 'DATE':
                data = data.reset_index()

            # save to local dir
            data_dir = os.path.join(output_dir, f"{index_name}_raw.csv")
            data.to_csv(data_dir)

            valid_list.append(index_name)
            
        except Exception as e:
            print(f"Failed to fetch data for {index_name}: {e}")
    
    # save the index for further processing
    save_index(valid_list, "0_macro_index")



def fetch_data(stock=True,market=True,macro=True):
    if stock:
        fetch_stock()
    if market:
        fetch_market_indices()
    if macro:
        fetch_macro_indices()

if __name__ == "__main__":
    fetch_data(True, True, True)
