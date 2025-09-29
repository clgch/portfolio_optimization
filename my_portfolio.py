import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

from skfolio import RatioMeasure, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.distribution import VineCopula
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.moments import (
    DenoiseCovariance,
    DetoneCovariance,
    GerberCovariance,
    ShrunkMu,
)
from skfolio.optimization import (
    MeanRisk,
    HierarchicalRiskParity,
    NestedClustersOptimization,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.pre_selection import SelectKExtremes
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import (
    BlackLitterman,
    EmpiricalPrior,
    EntropyPooling,
    FactorModel,
    OpinionPooling,
    SyntheticData,
 )
from skfolio.uncertainty_set import BootstrapMuUncertaintySet
from skfolio.prior import EmpiricalPrior

tickers = ["PSP5.PA", "MFEC.PA", "ETSZ.DE"] 

ohlc = yf.download(tickers, period="max")
prices = ohlc["Close"].dropna(how="all")
print(prices.tail())

prices_cleaned = prices[prices.index >= "2020-01-01"].interpolate()
#prices_cleaned.plot(figsize=(15,10))
#plt.show()

X = prices_to_returns(prices_cleaned)
#X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

X_train = X[X.index <= "2023-09-20"]
X_test = X[X.index >= "2023-09-20"]

management_fees = np.array([0.2, 0.12, 0.15])/(252 * 100)

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,
    management_fees=management_fees,
    risk_free_rate=0.0266/252
)

emp_prior = EmpiricalPrior(is_log_normal=True, investment_horizon=252)
emp_prior.fit(X_train)

np.random.seed(0)

B = 1000
n = len(X_train)
ess = 0

annualized_expected_returns = []
max_drawdown = []

for b in range(B):
    print(f"b = {b}")
    sample_id = np.random.choice(np.arange(n), size=n, replace=True)
    X_boot = X_train.iloc[sample_id]
    
    emp_prior = EmpiricalPrior(is_log_normal=True, investment_horizon=252)
    emp_prior.fit(X_boot)

    expected_returns = emp_prior.return_distribution_.mu

    if np.any(expected_returns > 0.0266):

        model.fit(X_boot)
        print(model.weights_)

        portfolio = model.predict(X_test)
        print(f"Annualized expected portfolio returns {portfolio.annualized_mean * 100} %")
        print(f"Annualized portfolio vol {portfolio.max_drawdown * 100} %")

        annualized_expected_returns.append(portfolio.annualized_mean * 100)
        max_drawdown.append(portfolio.max_drawdown * 100)

        ess += 1

print(f"ess = {ess}")

plt.hist(annualized_expected_returns, density=True)
plt.xlabel("Expected portfolio returns (%)")
plt.show()

plt.hist(max_drawdown, density=True)
plt.xlabel("Max drawdown (%)")
plt.show()


