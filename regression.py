import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LarsCV, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# trained with 7 linear regressor + 4 non-linear regressor (with Cross-Validition if needed)
Enable_LinearRegression = True
Enable_Ridge = True
Enable_Lasso = True
Enable_ElasticNet = True
Enable_Lars = True
Enable_BayesianRidge = True
Enable_HuberRegressor = True
Enable_RandomForestRegressor = True
Enable_KernelRidge = True
Enable_XGBRegressor = True
Enable_GPRegressor = True

# choose variables
task = "task2"
train_percentage = 0.8
n_jobs = -1

# load dataframe form csv
df = pd.read_csv(f"./data/{task}_train.csv")
if task == "task1":
    X_df = df[[f"x_{i}" for i in range(1, 11)]]
elif task == "task2":
    X_df = df[["x"]]
elif task == "task3":
    X_df = df[[f"x{i}" for i in range(1, 10)]]
y_df = df[["value"]].values.ravel()

# split dataframe into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, train_size = train_percentage, shuffle = True)

# regressor candidates
candidates = []
if Enable_LinearRegression == True:
    candidates.append(("LinearRegression", LinearRegression(n_jobs = n_jobs)))
if Enable_Ridge == True:
    candidates.append(("Ridge", Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas = np.logspace(-6, 6, 13)))])))
if Enable_Lasso == True:
    candidates.append(("Lasso", Pipeline([('scaler', StandardScaler()), ('lasso', LassoCV(alphas = np.logspace(-6, 6, 13), n_jobs = n_jobs))])))
if Enable_ElasticNet == True:
    candidates.append(("ElasticNet", Pipeline([('scaler', StandardScaler()), ('elasticnet', ElasticNetCV(alphas = np.logspace(-6, 6, 13), l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0], n_jobs = n_jobs))])))
if Enable_Lars == True:
    candidates.append(("Lars", LarsCV(cv = 5, n_jobs = n_jobs)))
if Enable_BayesianRidge == True:
    candidates.append(("BayesianRidge", BayesianRidge()))
if Enable_HuberRegressor == True:
    HuberRegressorCV = GridSearchCV(
        estimator = Pipeline([('scaler', StandardScaler()), ('huber', HuberRegressor())]),
        param_grid = {
            "huber__epsilon": [1.1, 1.35, 1.5, 2.0]
        },
        scoring = "neg_mean_squared_error",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("HuberRegressor", HuberRegressorCV))
if Enable_RandomForestRegressor == True:
    RandomForestRegressorCV = GridSearchCV(
        estimator = RandomForestRegressor(),
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10]
        },
        scoring = "neg_mean_squared_error",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("RandomForestRegressor", RandomForestRegressorCV))
if Enable_KernelRidge == True:
    KernelRidgeCV = GridSearchCV(
        estimator = KernelRidge(),
        param_grid = {
            "alpha": [0.1, 1.0, 10.0],
            "kernel": ['rbf'],
            "gamma": [0.001, 0.01, 0.1]
        },
        scoring = "neg_mean_squared_error",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("KernelRidge", KernelRidgeCV))
if Enable_XGBRegressor == True:
    XGBRegressorCV = GridSearchCV(
        estimator = xgb.XGBRegressor(objective = "reg:squarederror"),
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, None],
            'learning_rate': [0.05, 0.1],
        },
        scoring = "neg_mean_squared_error",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("XGBRegressor", XGBRegressorCV))
if Enable_GPRegressor == True:
    candidates.append(("GPRegressor", Pipeline([("scaler", StandardScaler()), ('gpregressor', GaussianProcessRegressor(kernel = C(1.0) * RBF(1.0, ((1e-7, 1e5))), n_restarts_optimizer = 10))])))

# train candidates with train data
with parallel_backend("threading", prefer = "threads"):
    for candidate in candidates:
        print(f"Training with {candidate[0]}...")
        candidate[1].fit(X = X_train, y = y_train)

# calculate MSE with remaining test data
min_mse = float("inf")
president = None
print(f"{"=" * 50}\nModel Name{" " * 37}MSE\n{"=" * 50}")
for candidate in candidates:
    mse = mean_squared_error(y_true = y_test, y_pred = candidate[1].predict(X_test))
    print(f"{candidate[0]:<33}{mse:.15f}")
    if mse < min_mse:
        min_mse = mse
        president = candidate
print(f"{"=" * 50}\nBest Model: {president[0]}\nMSE = {min_mse}\n{"=" * 50}")

# use president to pridect final output
df = pd.read_csv(f"./data/{task}_test.csv")
if task == "task1":
    X_df = df[[f"x_{i}" for i in range(1, 11)]]
elif task == "task2":
    X_df = df[["x"]]
else:
    X_df = df[[f"x{i}" for i in range(1, 10)]]
y_df = president[1].predict(X_df)
output_df = pd.DataFrame({"id": df["id"], "value": y_df})
output_df.to_csv(f"./output/{task}_{president[0]}_{min_mse:.6f}.csv", index = False)
print("finished output.")