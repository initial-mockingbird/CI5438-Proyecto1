from src.regression import train,linear_regression
from src.graph_utils import plot_lines, stacked_bar, plot_two_lines
from src.random_sample import testing_data,f,observations
import random
import numpy as np
import pandas as pd

def run_all():
  configs =\
    [ "config.yaml"
    , "make_year_kilo_seating_fuel_location.yaml" 
    , "no_categorics.yaml"
    ]

  
  def pad(n,ys):
    return ys + [ys[-1] for _ in range(n - len(ys))]

  results = [train(f"./config/{f}") for f in configs]

  x = max(list(map(lambda rs: rs[0],results)),key=len)
  errors = list(map(lambda rs: pad(len(x),rs[1].tolist()),results))
  r2s = list(map(lambda rs: rs[2],results))
  coefficients =  list(map(lambda rs: rs[3],results))
  plot_lines(x,errors,"Iteracion vs mse","Iteracion","mse",(3,1),configs)
  stacked_bar(r2s,["mediumpurple"],configs)

  print(r2s)

def run_all_test():
  configs =\
    [ "config.yaml"
    , "make_year_kilo_seating_fuel_location.yaml" 
    , "no_categorics.yaml"
    ]

  (x,errors,coefficients) = linear_regression(testing_data,"ys",max_iter=3e4)
  plot_lines(x,errors,"Iteracion vs mse","Iteracion","mse",(1,1),"f= 1 + 20w1 + 50w2")

  random.seed(10)
  w0s_Test = [1 for _ in range(observations)]
  w1s_Test = [random.uniform(0,1) for _ in range(observations)]
  w2s_Test = [random.uniform(0,1) for _ in range(observations)]
  Y_test = np.array([f(x,y) for (x,y) in zip(w1s_Test,w2s_Test)])

  X_test = pd.DataFrame({"w0":w0s_Test,"w1":w1s_Test,"w2":w2s_Test}).to_numpy()
  P = (np.matmul(X_test, coefficients))
  rss = sum(np.square(Y_test - P))
  tss = sum(np.square(Y_test - np.average(Y_test)))
  r2_score = 1 - rss/tss
  print(r2_score)
  print(coefficients)
  predicted = lambda x,y: coefficients[0] + coefficients[1]*x + coefficients[2]*y
  plot_two_lines(f,predicted)