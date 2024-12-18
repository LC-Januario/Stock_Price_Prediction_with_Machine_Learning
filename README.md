# Stock Price Prediction with XGBoost

This project aims to predict the opening and maximum prices of a stock for the current day using historical price data from the last 50 days. The prediction is made using the XGBoost machine learning algorithm. 

```markdown
# Stock Price Prediction and Analysis

This repository contains a Python notebook that analyzes historical stock prices and predicts future prices using machine learning models. The main functionality is implemented in a collaborative notebook and uses various libraries to process data, train models, and evaluate the results.

## Overview

### Main Block Execution
The main block of the notebook executes the `test_algorithm()` function and prints the results. The results include:

- Number of stocks analyzed
- Number of 'BUY' signals over the analyzed period
- Return on Investment (ROI) in percentage
- Number of profitable trades ('Greens')
- Number of unprofitable trades ('Reds')
- Green/Red Ratio
- DataFrames with their names and ROI

### Example Output
```python
# Print results for verification (positive ROI indicates profit)
print("Test Results: ")
print(f"Number of Stocks Analyzed: {len(output_dfs)}")
print(f"Number of 'BUY' Signals Over Analyzed Period: {len(signal_results)}")
print(f"ROI: {np.mean(signal_results):.2f}%")
print(f'Number of Greens: {len([x for x in signal_results if x > 0])}')
print(f'Number of Reds: {len([x for x in signal_results if x < 0])}')
print(f'Green/Red Ratio: {(len([x for x in signal_results if x > 0])/len([x for x in signal_results if x < 0])):.2f}')
print('')
print('')

# Print names of dataframes for verification
for name, df in output_dfs.items():
    print(f"DataFrame name: {name}")
    print(f'ROI: {np.sum(results_per_stock[name]):.2f}%')
    print(df.head(10))
    print('')
```

### Results
The results so far are:

- Number of Stocks Analyzed: 10
- Number of 'BUY' Signals Over Analyzed Period: 83
- ROI: 0.43%
- Number of Greens: 52
- Number of Reds: 31
- Green/Red Ratio: 1.68

## Functions

### `test_algorithm()`
This function processes multiple datasets from Google Drive and executes the `test_prediction_open_and_close_prices(file_path)` function for each dataset. The datasets include:

- HistoricalData_Netflix
- HistoricalData_AMD
- HistoricalData_Tesla
- HistoricalData_Amazon
- HistoricalData_Meta
- HistoricalData_Qualcomm
- HistoricalData_Cisco
- HistoricalData_Microsoft
- HistoricalData_Starbucks
- HistoricalData_Apple

The outputs of this function are `output_dataframes`, `signal_results`, and `results_per_stock`.

### `test_prediction_open_and_close_prices(file_path)`
This function processes CSV files containing daily stock price information. It performs the following steps:

1. Creates columns for each date:
    - Moving averages for 5, 10, and 20 days
    - Exponential moving averages (EMA) for 10, 20, and 50 days
    - Volatility for 5, 10, and 20 days

2. Trains three XGBoost models to predict:
    - Opening prices
    - Closing prices
    - Highest prices

The predictions are stored in a DataFrame called `predicao_day` and returned.

### Example of Data Processing
```python
daily_df['avg_5_days_close'] = daily_df['Close/Last'].rolling(window=5).mean()
daily_df['avg_5_days_volume'] = daily_df['Volume'].rolling(window=5).mean()
daily_df['avg_5_days_open'] = daily_df['Open'].rolling(window=5).mean()
daily_df['avg_5_days_high'] = daily_df['High'].rolling(window=5).mean()
daily_df['avg_5_days_low'] = daily_df['Low'].rolling(window=5).mean()

daily_df['avg_20_days_close'] = daily_df['Close/Last'].rolling(window=20).mean()
daily_df['avg_20_days_volume'] = daily_df['Volume'].rolling(window=20).mean()
daily_df['avg_20_days_open'] = daily_df['Open'].rolling(window=20).mean()
daily_df['avg_20_days_high'] = daily_df['High'].rolling(window=20).mean()
daily_df['avg_20_days_low'] = daily_df['Low'].rolling(window=20).mean()

daily_df['ema_10_days_close'] = daily_df['Close/Last'].ewm(span=10, adjust=False).mean()
daily_df['ema_20_days_close'] = daily_df['Close/Last'].ewm(span=20, adjust=False).mean()
daily_df['ema_50_days_close'] = daily_df['Close/Last'].ewm(span=50, adjust=False).mean()
daily_df['ema_10_days_open'] = daily_df['Open'].ewm(span=10, adjust=False).mean()
daily_df['ema_20_days_open'] = daily_df['Open'].ewm(span=20, adjust=False).mean()
daily_df['ema_50_days_open'] = daily_df['Open'].ewm(span=50, adjust=False).mean()
daily_df['ema_10_days_high'] = daily_df['High'].ewm(span=10, adjust=False).mean()
daily_df['ema_20_days_high'] = daily_df['High'].ewm(span=20, adjust=False).mean()
daily_df['ema_50_days_high'] = daily_df['High'].ewm(span=50, adjust=False).mean()

daily_df['volatility_5_days'] = daily_df['Close/Last'].rolling(window=5).std()
daily_df['volatility_10_days'] = daily_df['Close/Last'].rolling(window=10).std()
daily_df['volatility_20_days'] = daily_df['Close/Last'].rolling(window=20).std()
```

## Strategy for 'BUY' Signals
The strategy considered for generating 'BUY' signals and trade orders and then calculating the resulting ROI for that day is as follows: The signal is 'BUY' if the predicted high of the stock is at least 2,5% higher then predicted opening price and the predicted closing price is higher then the predicted opening price. The buying price is always the opening price. The selling price is equal to the predicted high if the real high is equal or bigger than the predicted high, else the selling price is the real closing price. The ROI is the diference between buying price and selling price divided by the buying price. The respective code:
```python
if predicao_day.loc[i, 'Real_High'] >= predicao_day.loc[i, 'Predicted_High']:

    predicao_day.loc[i, 'Sell_Price'] = predicao_day.loc[i, 'Predicted_High']
    predicao_day.loc[i, 'ROI'] = (predicao_day.loc[i, 'Sell_Price'] - predicao_day.loc[i, 'Real_Open']) / predicao_day.loc[i, 'Real_Open'] * 100
else:
    predicao_day.loc[i, 'Sell_Price'] = predicao_day.loc[i, 'Real_Close']
    predicao_day.loc[i, 'ROI'] = (predicao_day.loc[i, 'Sell_Price'] - predicao_day.loc[i, 'Real_Open']) / predicao_day.loc[i, 'Real_Open'] * 100
  
if (predicao_day.loc[i, 'Predicted_High'] >= predicao_day.loc[i, 'Predicted_Open']*(1 + aimed_profit_margin)) and (predicao_day.loc[i, 'Predicted_Close'] > predicao_day.loc[i, 'Predicted_Open']):
      predicao_day.loc[i, 'Buy_Signal']= 'BUY'
else:
      predicao_day.loc[i, 'Buy_Signal'] = 'DONT
```

## Next Steps for Improvement
- **Scaling Data**: Implement a method to scale the data during the processing pipeline and then unscale the data at `predicao_day`. Scaling is expected to yield better results.
- **Parameter Optimization**: Fine-tune the parameters for the XGBoost models to improve accuracy.
- **Further Testing**: Although the positive ROI and results indicate the model is working well, the results are small enough that they could be due to chance. Further testing is necessary before using this model for anything beyond academic purposes.

## Requirements
- Python 3.x
- Libraries: NumPy, pandas, xgboost, sklearn

## How to Use
1. Upload the datasets to Google Drive.
2. Run the notebook to process the data and train the models.
3. Review the printed results to analyze the performance and predictions.

## Results Interpretation
- **ROI (Return on Investment)**: Indicates the percentage of profit or loss.
- **Green/Red Ratio**: Shows the ratio of profitable trades to unprofitable trades.


