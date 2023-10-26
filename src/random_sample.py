import random
import pandas as pd

random.seed(5)

observations = 10000

# w0 = 3
# w1 = 2
# w2 = 5

f = lambda x,y : 3 + 2 * x + 5 * y

w1s = [random.uniform(0,1) for _ in range(observations)]
w2s = [random.uniform(0,1) for _ in range(observations)]
ys = [f(x,y) for (x,y) in zip(w1s,w2s)]
df = pd.DataFrame(\
  { "w1s": w1s 
  , "w2s": w2s
  , "ys": ys 
  }
)
