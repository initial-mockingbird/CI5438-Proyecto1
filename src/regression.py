import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(data,objective,a=1e-3,max_iter=1e3, epsilon=1e-3):

  column_mapping = dict([ (name,f"w{i+1}") if name != objective else (objective,objective)  for (i,name) in enumerate(data.columns)])
  
  X_df = data.copy().rename(column_mapping).drop(objective,axis=1)
  X_df.insert(0,"w0",1)
  Y = data[objective].to_numpy()
  X = X_df.to_numpy()
  W = X_df.copy().head(1).to_numpy()[0]
  current_iteration = 0
  error = epsilon + 1
  m = len(Y)

  plt.style.use('_mpl-gallery')
  acc_errors = []


  while(current_iteration < max_iter and error > epsilon ):
    P   = (np.matmul(X, W))
    mse = (2/m) * (np.matmul(X.transpose(), P - Y))
    W = W - a * mse
    error = max(np.abs(mse)) #np.average(np.abs(mse))
    acc_errors.append(error)
    current_iteration += 1
  
  x = np.array([i for (i,_) in enumerate(acc_errors)])
  acc_errors = np.array(acc_errors)
  fig, ax = plt.subplots()

  ax.plot(x, acc_errors, linewidth=2.0)
  ax.title.set_text('Iteracion VS mse')
  plt.show()
  
  return W