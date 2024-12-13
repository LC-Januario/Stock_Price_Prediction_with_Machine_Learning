# Stock Price Prediction with XGBoost

This project aims to predict the opening and maximum prices of a stock for the current day using historical price data from the last 10 days. The prediction is made using the XGBoost machine learning algorithm. 

## Overview

The model takes in stock price data from the last 10 days as input and outputs the predicted opening and maximum prices for the current day. The predictions currently have a low accuracy, and the trading strategies based on these predictions have led to a loss of 2.43% at best. However, there is potential for improvement through further optimization.


## Data

The data consists of historical stock prices, including the opening, closing, high, low, and volume for each day. Ensure that your data is formatted correctly and includes the necessary features.


### Example

Here's a simplified example of how to use the script:

```python
# Test the prediction Algorithym
output_dfs, signal_results, results_avg = test_algorythim()

# Print results for verification (positive ROI indicates profit)
print("Test Results: ")
print(f"Number of Stocks Analyzed: {len(output_dfs)}")
print(f"Number of 'BUY' Signals Over Analyzed Period: {len(signal_results)}")
print(f"ROI: {np.mean(signal_results):.2f}%")  # Changed line: Directly use signal_results
print(f'Number of Greens: {len([x for x in signal_results if x > 0])}')
print(f'Number of Reds: {len([x for x in signal_results if x < 0])}')
print(f'Green/Red Ratio: {(len([x for x in signal_results if x > 0])/len([x for x in signal_results if x < 0])):.2f}')
print('')
print('')

for name, df in output_dfs.items():
    print(f"DataFrame name: {name}")
    print(f'ROI: {np.mean(results_avg[name]):.2f}%')
    print(df.head())
    print('')
```

## Current Performance

- **Accuracy:** Currently, the model has a low accuracy rating.
- **Trading Strategy Performance:** Trading strategies based on the model's predictions have yielded losses of 2.43% at best.

## Future Work

- **Model Optimization:** There is significant room for improvement through further optimization of the model parameters and feature engineering.
- **Additional Data:** Including more features and a larger dataset may enhance the model's performance.
- **Advanced Techniques:** Experimenting with other machine learning algorithms and techniques could lead to better predictions.

## Contributing

Contributions are welcome! If you have ideas for improving the model or the script, feel free to fork the repository and submit a pull request.
