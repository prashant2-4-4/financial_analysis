import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FinancialDataLoader:
    """Class to handle financial data collection and preprocessing"""
    
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.returns = {}
        self.features = {}
        
    def fetch_data(self):
        """Fetch historical price data for all symbols"""
        print("Fetching financial data...")
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                self.data[symbol] = data
                print(f"✓ Fetched data for {symbol}: {len(data)} records")
            except Exception as e:
                print(f"✗ Error fetching data for {symbol}: {str(e)}")
                
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for feature engineering"""
        # Simple Moving Averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        data['BB_Middle'] = data['Close'].rolling(window=bb_window).mean()
        bb_std = data['Close'].rolling(window=bb_window).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    
    def preprocess_data(self):
        """Preprocess data and calculate returns"""
        print("Preprocessing data and calculating features...")
        
        for symbol in self.symbols:
            if symbol in self.data:
                # Calculate technical indicators
                self.data[symbol] = self.calculate_technical_indicators(self.data[symbol])
                
                # Calculate returns
                self.returns[symbol] = self.data[symbol]['Close'].pct_change()
                
                # Prepare features
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                               'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                               'MACD', 'MACD_Signal', 'RSI', 'BB_Position', 
                               'BB_Width', 'Volatility', 'Volume_Ratio']
                
                self.features[symbol] = self.data[symbol][feature_cols].copy()
                
                # Add lagged returns as features
                for lag in range(1, 6):
                    self.features[symbol][f'Return_Lag_{lag}'] = self.returns[symbol].shift(lag)
                
                # Drop NaN values
                self.features[symbol] = self.features[symbol].dropna()
                self.returns[symbol] = self.returns[symbol].dropna()
                
                print(f"✓ Processed {symbol}: {len(self.features[symbol])} clean records")

class LSTMModel:
    """LSTM model for return forecasting"""
    
    def __init__(self, sequence_length=60, n_features=22):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, features, returns):
        """Prepare sequences for LSTM training"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(returns.iloc[i])
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the LSTM model"""
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

class TransformerModel:
    """Transformer model for return forecasting"""
    
    def __init__(self, sequence_length=60, n_features=22, d_model=128, n_heads=8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.model = None
        self.scaler = StandardScaler()
        
    def transformer_encoder(self, inputs):
        """Transformer encoder block"""
        # Multi-head attention
        attention = MultiHeadAttention(num_heads=self.n_heads, key_dim=self.d_model)(inputs, inputs)
        attention = Dropout(0.1)(attention)
        attention = LayerNormalization()(inputs + attention)
        
        # Feed forward network
        ffn = Dense(self.d_model * 2, activation='relu')(attention)
        ffn = Dense(self.d_model)(ffn)
        ffn = Dropout(0.1)(ffn)
        ffn = LayerNormalization()(attention + ffn)
        
        return ffn
    
    def build_model(self):
        """Build Transformer model architecture"""
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Project to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding (simplified)
        position_encoding = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_encoding = tf.expand_dims(position_encoding, 1)
        position_encoding = tf.cast(position_encoding, tf.float32)
        x += tf.sin(position_encoding / 10000.0)
        
        # Transformer encoder blocks
        x = self.transformer_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global average pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_sequences(self, features, returns):
        """Prepare sequences for Transformer training"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(returns.iloc[i])
            
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the Transformer model"""
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

class PortfolioOptimizer:
    """Dynamic portfolio optimization using model predictions"""
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.n_assets = len(symbols)
        
    def mean_reversion_weights(self, predictions, lookback=20):
        """Calculate portfolio weights using mean reversion strategy"""
        # Convert predictions to expected returns
        expected_returns = np.array(predictions).flatten()
        
        # Simple mean reversion: higher weight for assets with higher predicted returns
        weights = np.exp(expected_returns * 10)  # Amplify differences
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Apply constraints
        weights = np.clip(weights, 0.02, 0.4)  # Min 2%, Max 40% per asset
        weights = weights / np.sum(weights)  # Renormalize
        
        return weights
    
    def momentum_weights(self, predictions, historical_returns):
        """Calculate portfolio weights using momentum strategy"""
        expected_returns = np.array(predictions).flatten()
        
        # Combine predictions with historical momentum
        momentum = historical_returns.iloc[-20:].mean()  # 20-day momentum
        combined_signal = 0.7 * expected_returns + 0.3 * momentum.values
        
        # Calculate weights
        weights = np.maximum(combined_signal, 0)  # Long-only
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets
            
        return weights
    
    def risk_parity_weights(self, returns_data, window=60):
        """Calculate risk parity weights"""
        # Calculate covariance matrix
        recent_returns = returns_data.iloc[-window:]
        cov_matrix = recent_returns.cov()
        
        # Simple risk parity approximation
        volatilities = np.sqrt(np.diag(cov_matrix))
        weights = 1 / volatilities
        weights = weights / weights.sum()
        
        return weights

class PerformanceEvaluator:
    """Evaluate portfolio performance"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, returns, benchmark_returns=None):
        """Calculate comprehensive performance metrics"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': (returns > 0).mean(),
            'Best Day': returns.max(),
            'Worst Day': returns.min()
        }
        
        # Benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_returns = pd.Series(benchmark_returns) if not isinstance(benchmark_returns, pd.Series) else benchmark_returns
            excess_returns = returns - benchmark_returns
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            metrics['Information Ratio'] = information_ratio
            
        return metrics
    
    def calculate_sortino_ratio(self, returns, target_return=0):
        """Calculate Sortino ratio"""
        excess_returns = returns - target_return/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation > 0:
            return (returns.mean() * 252 - target_return) / downside_deviation
        else:
            return 0
    
    def plot_performance(self, returns, benchmark_returns=None, title="Portfolio Performance"):
        """Plot performance charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative.index, cumulative.values, label='Portfolio', linewidth=2)
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                           label='Benchmark', alpha=0.7)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 0].set_title('Rolling 1-Year Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Return distribution
        axes[1, 1].hist(returns.values, bins=50, alpha=0.7, density=True)
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Main execution pipeline
def main():
    """Main execution pipeline for the AI portfolio management system"""
    
    # Configuration
    SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'SPY']
    START_DATE = '2018-01-01'
    END_DATE = '2024-01-01'
    
    
    # Step 1: Data Collection and Preprocessing
    data_loader = FinancialDataLoader(SYMBOLS, START_DATE, END_DATE)
    data_loader.fetch_data()
    data_loader.preprocess_data()
    
    # Step 2: Model Training
    
    # Prepare combined dataset
    all_features = []
    all_returns = []
    
    for symbol in SYMBOLS:
        if symbol in data_loader.features and symbol in data_loader.returns:
            # Align features and returns
            common_index = data_loader.features[symbol].index.intersection(data_loader.returns[symbol].index)
            features_aligned = data_loader.features[symbol].loc[common_index]
            returns_aligned = data_loader.returns[symbol].loc[common_index]
            
            all_features.append(features_aligned)
            all_returns.append(returns_aligned)
    
    # Combine all data
    combined_features = pd.concat(all_features, axis=0, ignore_index=True)
    combined_returns = pd.concat(all_returns, axis=0, ignore_index=True)
    
    # Remove any remaining NaN values
    mask = ~(combined_features.isna().any(axis=1) | combined_returns.isna())
    combined_features = combined_features[mask]
    combined_returns = combined_returns[mask]
    
    print(f"Combined dataset shape: {combined_features.shape}")
    
    # Train LSTM Model
    lstm_model = LSTMModel(sequence_length=30, n_features=combined_features.shape[1])
    X_lstm, y_lstm = lstm_model.prepare_sequences(combined_features, combined_returns)
    
    # Split data
    split_idx = int(0.8 * len(X_lstm))
    X_train_lstm, X_val_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train_lstm, y_val_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
    
    lstm_model.build_model()
    lstm_history = lstm_model.train(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, epochs=50)
    
    # Train Transformer Model
    transformer_model = TransformerModel(sequence_length=30, n_features=combined_features.shape[1])
    X_transformer, y_transformer = transformer_model.prepare_sequences(combined_features, combined_returns)
    
    X_train_trans, X_val_trans = X_transformer[:split_idx], X_transformer[split_idx:]
    y_train_trans, y_val_trans = y_transformer[:split_idx], y_transformer[split_idx:]
    
    transformer_model.build_model()
    transformer_history = transformer_model.train(X_train_trans, y_train_trans, X_val_trans, y_val_trans, epochs=50)
    
    # Step 3: Portfolio Construction and Backtesting
    portfolio_symbols = [s for s in SYMBOLS if s != 'SPY']  # Remove SPY (use as benchmark)
    returns_df = pd.DataFrame({symbol: data_loader.returns[symbol] for symbol in portfolio_symbols})
    returns_df = returns_df.dropna()
    
    # Simple backtesting simulation
    backtest_period = returns_df.index[-252:]  # Last year
    portfolio_returns = []
    
    # Use equal weights for simplicity (in practice, use model predictions)
    weights = np.ones(len(portfolio_symbols)) / len(portfolio_symbols)
    
    for date in backtest_period:
        if date in returns_df.index:
            daily_returns = returns_df.loc[date].values
            portfolio_return = np.dot(weights, daily_returns)
            portfolio_returns.append(portfolio_return)
    
    portfolio_returns = pd.Series(portfolio_returns, index=backtest_period[:len(portfolio_returns)])
    
    # Benchmark returns (SPY)
    benchmark_returns = data_loader.returns['SPY'].loc[portfolio_returns.index]
    
    # Step 4: Performance Evaluation
    evaluator = PerformanceEvaluator()
    
    # Calculate metrics
    portfolio_metrics = evaluator.calculate_metrics(portfolio_returns, benchmark_returns)
    benchmark_metrics = evaluator.calculate_metrics(benchmark_returns)
    
    for metric, value in portfolio_metrics.items():
        if isinstance(value, float):
            print(f"{metric:<20}: {value:>10.4f}")
        else:
            print(f"{metric:<20}: {value:>10}")
    
    for metric, value in benchmark_metrics.items():
        if isinstance(value, float):
            print(f"{metric:<20}: {value:>10.4f}")
        else:
            print(f"{metric:<20}: {value:>10}")
    
    # Plot results
    evaluator.plot_performance(portfolio_returns, benchmark_returns, 
                              "AI Portfolio vs SPY Benchmark")
    
    
    return {
        'data_loader': data_loader,
        'lstm_model': lstm_model,
        'transformer_model': transformer_model,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'portfolio_metrics': portfolio_metrics,
        'benchmark_metrics': benchmark_metrics
    }

if __name__ == "__main__":
    # pip install yfinance tensorflow scikit-learn matplotlib seaborn pandas numpy
    results = main()
