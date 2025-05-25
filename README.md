This document provides instructions on how to set up the environment and run the AI-driven quantitative finance project.
1. Prerequisites:
Python: Ensure you have Python 3.7 or higher installed.

2. Setup Virtual Environment:

conda activate my_env

Bash
3. Install Dependencies:
pip install numpy pandas yfinance matplotlib seaborn scikit-learn tensorflow

Bash
4. Obtain the Code:
Save the provided Python script to a file.

5. Running the Script:

python financial_analysis.py

Bash
6. Expected Output:
Console Output: The script will print progress messages to the console, including:
Data fetching status for each symbol.
Data preprocessing logs.
Model training progress (epochs and loss/metrics for LSTM and Transformer models).
Portfolio performance metrics for both the constructed portfolio and the benchmark (SPY).
Plots: Matplotlib windows will pop up displaying performance charts:
Cumulative Returns (Portfolio vs. Benchmark).
Portfolio Drawdown.
Rolling 1-Year Sharpe Ratio.
Return Distribution.
The script may take some time to run, especially during model training, depending on your hardware and the date range specified.


7. Data:
The script uses the yfinance library to automatically download historical financial data for the specified stock symbols (AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, SPY) and date range (2018-01-01 to 2024-01-01 by default).
No manual data file download is required to run the base script.
