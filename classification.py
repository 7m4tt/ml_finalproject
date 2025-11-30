import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# trained with 7 classifier (with Cross-Validition if needed)
Enable_LogisticRegression = True
Enable_KNeighborsClassifier = True
Enable_SVC = True
Enable_DecisionTreeClassifier = True
Enable_RandomForestClassifier = True
Enable_XGBClassifier = True
Enable_GradientBoostingClassifier = True

# choose variables
task = "task5"
train_percentage = 0.8
n_jobs = -1

# load dataframe form csv
df = pd.read_csv(f"./data/{task}_train.csv")
if task == "task4":
    X_df = df[[f"x_{i}" for i in range(1, 11)]]
elif task == "task5":
    X_df = df[[f"x_{i}" for i in range(1, 21)]]
le = LabelEncoder()
y_df = le.fit_transform(df[["label"]].values.ravel())

# split dataframe into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, train_size = train_percentage, shuffle = True)

# regressor candidates
candidates = []
if Enable_LogisticRegression == True:
    LR_CV = GridSearchCV(
        estimator = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(solver = "lbfgs"))]),
        param_grid = {
            "lr__C": np.logspace(-3, 3, 7),
        },
        scoring = "accuracy",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("LogisticRegression", LR_CV))

if Enable_KNeighborsClassifier == True:
    KNN_CV = GridSearchCV(
        estimator = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
        param_grid = {
            "knn__n_neighbors": [3, 5, 7, 9],
        },
        scoring = "accuracy",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("KNeighborsClassifier", KNN_CV))

if Enable_SVC == True:
    SVC_CV = GridSearchCV(
        estimator = Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
        param_grid = {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["rbf", "linear"],
        },
        scoring = "accuracy",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("SVC", SVC_CV))

if Enable_DecisionTreeClassifier == True:
    candidates.append(("DecisionTreeClassifier", DecisionTreeClassifier()))

if Enable_RandomForestClassifier == True:
    RandomForestClassifierCV = GridSearchCV(
        estimator = RandomForestClassifier(),
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
        },
        scoring = "accuracy",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("RandomForestClassifier", RandomForestClassifierCV))

if Enable_XGBClassifier == True:
    XGBClassifierCV = GridSearchCV(
        estimator = xgb.XGBClassifier(objective = "multi:softprob", eval_metric = "mlogloss"),
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, None],
            "learning_rate": [0.05, 0.1],
        },
        scoring = "accuracy",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("XGBClassifier", XGBClassifierCV))

if Enable_GradientBoostingClassifier == True:
    GradientBoostingClassifierCV = GridSearchCV(
        estimator = GradientBoostingClassifier(),
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
        },
        scoring = "accuracy",
        cv = 5,
        n_jobs = n_jobs
    )
    candidates.append(("GradientBoostingClassifier", GradientBoostingClassifierCV))

# train candidates with train data
with parallel_backend("threading", prefer = "threads"):
    for candidate in candidates:
        print(f"Training with {candidate[0]}...")
        candidate[1].fit(X = X_train, y = y_train)

# calculate accuracy with remaining test data
max_acc = 0.0
president = None
print(f"{"=" * 39}\nModel Name{" " * 21}Accuracy\n{"=" * 39}")
for candidate in candidates:
    acc = accuracy_score(y_true = y_test, y_pred = candidate[1].predict(X_test))
    print(f"{candidate[0]:<33}{acc:.4f}")
    if acc > max_acc:
        max_acc = acc
        president = candidate
print(f"{"=" * 39}\nBest Model: {president[0]}\nAccuracy = {max_acc:.4f}\n{"=" * 39}")

# use president to pridect final output
df = pd.read_csv(f"./data/{task}_test.csv")
if task == "task4":
    X_df = df[[f"x_{i}" for i in range(1, 11)]]
elif task == "task5":
    X_df = df[[f"x_{i}" for i in range(1, 21)]]
y_df = le.inverse_transform(president[1].predict(X_df))
output_df = pd.DataFrame({"id": df["id"], "value": y_df})
output_df.to_csv(f"./output/{task}_{president[0]}_{max_acc}.csv", index = False)
print("finished output.")