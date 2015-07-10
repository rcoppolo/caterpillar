import os
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

# scoring function
# https://www.kaggle.com/jpopham91/caterpillar-tube-pricing/rmlse-vectorized
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

if __name__ == "__main__":

  train = pd.read_csv('./downloaded/train_set.csv', parse_dates=[2,])
  test = pd.read_csv('./downloaded/test_set.csv', parse_dates=[3,])
  labels = train.cost.values
  ids = test.id.values.astype(int)

  suppliers = pd.get_dummies(pd.concat([train.supplier, test.supplier]))
  features = np.column_stack((suppliers.values[0:train.shape[0]], pd.get_dummies(train.bracket_pricing),
    train.quote_date.dt.year, train.quote_date.dt.month, train.quote_date.dt.dayofyear, train.quote_date.dt.dayofweek,
    train.quote_date.dt.day, train.quantity))
  test_features = np.column_stack((suppliers.values[train.shape[0]-1:-1], pd.get_dummies(test.bracket_pricing),
    test.quote_date.dt.year, test.quote_date.dt.month, test.quote_date.dt.dayofyear, test.quote_date.dt.dayofweek,
    test.quote_date.dt.day, test.quantity))

  cvs = 5
  test_split = 0.3
  tube_assembly_ids = np.unique(train.tube_assembly_id)
  results = []

  for _i in range(cvs):
    print("Cross validating...")
    random = np.random.rand(tube_assembly_ids.shape[0])
    train_mask = np.in1d(train.tube_assembly_id, tube_assembly_ids[np.where(random > test_split)])
    test_mask = np.in1d(train.tube_assembly_id, tube_assembly_ids[np.where(random <= test_split)])

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = rmsle(y_test, preds)
    print(result)
    results.append(result)

  print("Average score: %.6f" % np.mean(results))

  if "--submit" in sys.argv:

    model.fit(features, labels)
    predictions = model.predict(test_features)

    # write submission file w/ git hash identifier
    submission = pd.DataFrame({"id": ids, "cost": predictions})
    proc = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    submission_name = proc.stdout.read().strip()
    submission.to_csv("./submissions/" + submission_name + ".csv", index=False)
