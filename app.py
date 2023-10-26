from src.config.reader import read_data
from src.regression import linear_regression
from math import floor
import numpy as np
import yaml


def main():
  info = read_data()
  data = info["data"]
  train=data.sample(frac=info["split"]["training"],random_state=200)
  test=data.drop(train.index)
  coefficients = linear_regression(train,info["target"],max_iter=1e3)
  Y_test = test[info["target"]].to_numpy()
  X_test = test.copy().drop(info["target"],axis=1)
  X_test.insert(0,"w0",1)
  X_test = X_test.to_numpy()
  m = len(Y_test)
  P = (np.matmul(X_test, coefficients))
  rss = sum(np.square(P - Y_test))
  tss = sum(np.square(Y_test - np.average(P)))
  r2_score = 1 - rss/tss
  print(coefficients)
  print(r2_score)

if (__name__=="__main__"):
  main()