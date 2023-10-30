from src.config.reader import read_data
import pandas as pd
import numpy as np


def linear_regression(data,objective,a=1e-3,max_iter=1e3, epsilon=2e-3):

  # remapeamos los nombres para poder incluir "w0" sin clashes
  column_mapping = dict([ (name,f"w{i+1}") if name != objective else (objective,objective)  for (i,name) in enumerate(data.columns)])
  
  # construimos: X,Y,W_0
  X_df = data.copy().rename(column_mapping).drop(objective,axis=1)
  X_df.insert(0,"w0",1)
  Y = data[objective].to_numpy()
  X = X_df.to_numpy()
  W = X_df.copy().head(1).to_numpy()[0]

  # seteando valores iniciales del loop y constantes
  current_iteration = 0
  error = epsilon + 1
  m = len(Y)

  # Para plotear los errores
  acc_errors = []

  # metodo iterativo
  while(current_iteration < max_iter and error > epsilon ):
    P   = X @ W
    delta = P - Y
    W = W - a/m * (X.transpose() @ delta)
    error = abs(1/(2*m) * np.transpose(delta) @ delta) #np.average(np.abs(mse))
    acc_errors.append(error)
    current_iteration += 1
  
  x = np.array([i for (i,_) in enumerate(acc_errors)])
  acc_errors = np.array(acc_errors)
  
  return (x,acc_errors,W)


def train(conf_location):
  info = read_data(conf_location)
  data = info["data"]
  train=data.sample(frac=info["split"]["training"],random_state=200)
  test=data.drop(train.index)
  (x,acc_errors,coefficients) = linear_regression(train,info["target"],max_iter=1e5)

  Y_test = test[info["target"]].to_numpy()
  X_test = test.copy().drop(info["target"],axis=1)
  X_test.insert(0,"w0",1)
  X_test = X_test.to_numpy()

  P = (np.matmul(X_test, coefficients))
  rss = sum(np.square(Y_test - P))
  tss = sum(np.square(Y_test - np.average(Y_test)))
  r2_score = 1 - rss/tss
  return (x,acc_errors,r2_score,coefficients)
