# Multi-Stock Prediction and Market-Regime Analysis

This project builds an end-to-end machine-learning workflow for analyzing historical stock-market data. It transforms daily OHLCV data into technical indicators, creates forward-looking classification and volatility targets, compares supervised-learning models, and applies unsupervised learning to identify latent market regimes.

The dataset contains 11 U.S. equities and financial assets:

`AMD`, `GLD`, `GS`, `INTC`, `JPM`, `META`, `MSFT`, `MU`, `NVDA`, `RXRX`, and `TSLA`.

## Project Goals

- Engineer trend, momentum, volatility, and volume-based predictors from daily market data.
- Predict whether a stock's closing price will rise by at least 2% within the next five trading days.
- Predict future 20-day volatility as a continuous outcome.
- Compare linear, tree-based, ensemble, and neural-network models.
- Use PCA, K-Means, and Gaussian Mixture Models to explore market structure and regimes.

## Repository Structure

```text
.
├── Final_Project data_preprocessing.ipynb  # Feature and target engineering
├── Prediction_Part.ipynb                   # EDA, supervised learning, and clustering
├── Pure_data/                              # Raw OHLCV files for 11 symbols
├── all_stocks_with_features.csv            # Combined engineered feature dataset
├── all_stocks_targets.csv                  # Combined forward-looking target dataset
└── README.md
```

## Data

Each raw CSV contains daily:

- Date
- Open, high, low, and closing prices
- Trading volume
- Stock symbol

The available history varies by asset. In the supplied files, the earliest observation is March 17, 1980, and the latest is February 11, 2026. RXRX has the shortest history, beginning in April 2021.

The combined feature file contains 82,754 observations. After removing incomplete rolling-window observations, the modeling notebook uses approximately 80,000 complete samples and 39 engineered features.

## Feature Engineering

The preprocessing notebook creates the following groups of technical indicators:

| Category | Features |
| --- | --- |
| Trend | EMA (5, 10, 20, 50 days); SMA (50, 100, 200 days) |
| Momentum | RSI (5, 14, 21 days); MACD, signal line, histogram; stochastic %K and %D (5 and 14 days) |
| Volatility | ATR (5, 14, 21 days); 20-day Bollinger Bands |
| Volume | Volume SMA (5, 20, 60 days); OBV; OBV Z-scores (20 and 50 days); MFI (5, 14, 21 days) |
| Historical returns | 1, 5, 10, 20, 50, 100, and 250-day returns |

## Prediction Targets

The project generates 104 forward-looking targets:

- **High-reach targets:** whether a future intraday high reaches a specified gain.
- **Low-reach targets:** whether a future intraday low reaches a specified loss.
- **Close-up targets:** whether a future closing price reaches a specified gain.
- **Close-down targets:** whether a future closing price reaches a specified loss.
- **Volatility targets:** realized future volatility over 5, 20, 100, and 250 trading days.

Price targets combine five forecast horizons (`1`, `5`, `20`, `100`, and `250` days) with five thresholds (`2%`, `5%`, `10%`, `20%`, and `50%`).

## Modeling Workflow

### Binary Classification

The primary classification task predicts:

```text
price_target_close_up_5d_2pct
```

This label equals 1 when the closing price reaches at least 2% above its current value within the next five trading days, and 0 otherwise.

Models:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Volatility Regression

The regression task predicts 20-day forward volatility:

```text
volatility_target_20d
```

Models:

- Random Forest Regressor with grid-search tuning
- Multilayer Perceptron with hidden layers of 128, 64, and 32 neurons

### Unsupervised Learning

- PCA for dimensionality reduction and feature interpretation
- K-Means for hard market-regime assignments
- Gaussian Mixture Model for probabilistic regime assignments
- Silhouette score, AIC, BIC, Adjusted Rand Index, and Normalized Mutual Information for evaluation and comparison

## Results

The following values are the results stored or described in the supplied modeling notebook.

### Five-Day Price Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.691 | 0.689 | 0.698 | 0.694 | 0.755 |
| Random Forest | **0.729** | **0.744** | 0.699 | 0.721 | **0.808** |
| Gradient Boosting | 0.728 | 0.733 | **0.718** | **0.725** | 0.805 |

Tree-based ensembles outperform the linear baseline, indicating that nonlinear interactions among returns, momentum indicators, and volatility measures contain useful predictive information. Multi-horizon return features are among the most influential predictors.

### Twenty-Day Volatility Regression

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Random Forest Regressor | 0.00951 | 0.00648 | 0.618 |
| MLP Neural Network* | ~0.0076 | ~0.0051 | ~0.75 |

\*The Random Forest metrics are stored as executed output. The MLP values are reported in the notebook's written interpretation and should be reproduced by rerunning the notebook.

### Dimensionality Reduction and Clustering

- PC1 and PC2 explain approximately 32.7% and 28.6% of total variance.
- Ten principal components retain approximately 90% of total variance; thirteen retain approximately 95%.
- K-Means with four clusters is used to describe bullish, bearish/consolidation, high-volatility, and low-volatility regimes.
- A four-component GMM provides soft assignments and highlights observations near regime boundaries.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Usage

Start Jupyter:

```bash
jupyter lab
```

Run the notebooks in this order:

1. `Final_Project data_preprocessing.ipynb`
2. `Prediction_Part.ipynb`

The first notebook reads the files in `Pure_data/` and creates the combined feature and target CSVs. The second notebook loads those outputs, performs exploratory analysis, trains the supervised models, and runs the unsupervised-learning workflow.

Because the generated CSV files are large, use Git LFS if they will be committed to a Git repository:

```bash
git lfs track "*.csv"
```

## Reproducibility Notes and Limitations

- The supplied feature and target files have different row counts: 82,754 and 82,765. The current modeling notebook assigns targets to features by row position. For reliable evaluation, retain `Date` and `symbol` in both outputs and join on those keys before modeling.
- The current random train/test split mixes observations from different dates. Because adjacent market observations are correlated, this can overestimate out-of-sample performance. A chronological split, walk-forward validation, or `TimeSeriesSplit` is recommended.
- Hyperparameter tuning for the volatility model currently uses ordinary K-fold cross-validation. Time-aware cross-validation would better represent real forecasting conditions.
- Technical indicators are derived only from historical price and volume. They do not include fundamentals, news, macroeconomic variables, transaction costs, slippage, or market-impact effects.
- The clustering labels are analytical interpretations, not objectively observed market regimes.

## Possible Improvements

- Build features and targets in a single keyed table to prevent row-alignment errors.
- Add walk-forward backtesting and compare against simple buy-and-hold or majority-class baselines.
- Evaluate performance separately by stock and by market period.
- Tune probability thresholds using trading objectives rather than accuracy alone.
- Add SHAP-based interpretation, probability calibration, and confidence intervals.
- Incorporate fundamental, macroeconomic, options, and news-sentiment data.
- Measure strategy performance after transaction costs using Sharpe ratio, maximum drawdown, and turnover.

## Disclaimer

This project is for educational and research purposes only. Its outputs are not financial advice and should not be used as the sole basis for investment decisions.
