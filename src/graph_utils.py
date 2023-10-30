import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_two_lines(reference,prediction):
  plt.style.use('seaborn-darkgrid')

  xs = np.linspace(0, 1, 100)
  ys = np.linspace(0, 1, 100)

  X, Y = np.meshgrid(xs, ys)

  @np.vectorize
  def reference_v(x,y):
    return reference(x,y)

  @np.vectorize
  def prediction_v(x,y):
    return prediction(x,y)

  Z_reference = reference_v(X,Y)
  Z_prediction = prediction_v(X,Y)

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.plot_surface(X, Y, Z_reference,color="lightseagreen",label="Reference: 1 + 20w1 + 50w2")
  ax.plot_surface(X, Y, Z_prediction,color="mediumpurple",label="Reference: 5.45 + 17.05w1 + 44.54w2")

  ax.set_xlabel("w1")
  ax.set_ylabel("w2")
  ax.set_zlabel("z")
  fakeRefline = mpl.lines.Line2D([0],[0], linestyle="none", c='lightseagreen', marker = 'o')
  fakePredline = mpl.lines.Line2D([0],[0], linestyle="none", c='mediumpurple', marker = 'o')
  ax.legend([fakeRefline, fakePredline], ['Reference: 1 + 20w1 + 50w2','Reference: 5.45 + 17.05w1 + 44.54w2'], numpoints = 1)
  plt.show()


def plot_lines(x,ys,title,xlabel,ylabel,dims,titles):
  plt.style.use('seaborn-darkgrid')
  fig, ax = plt.subplots(*dims)

  iy = 0

  if (dims[0] > 1 and dims[1] > 1):
    for i in range(dims[0]):
      for j in range(dims[1]):
        if iy >= len(ys):
          break
        y = ys[iy]
        ax[i,j].plot(x, y, marker='', color='blueviolet', linewidth=4, alpha=0.7)
        ax[i,j].set_title(titles[iy],fontsize=12, fontweight=1)
        ax[i,j].set_ylabel(ylabel)
        ax[i,j].set_xlabel(xlabel)
        iy += 1
  elif (dims[0] == 1 and dims[1] == 1):
    y = ys
    ax.plot(x, y, marker='', color='blueviolet', linewidth=4, alpha=0.7)
    ax.set_title(titles,fontsize=12, fontweight=1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
  else:
    for i in range(dims[0]*dims[1]):
      if iy >= len(ys):
        break
      y = ys[iy]
      ax[i].plot(x, y, marker='', color='blueviolet', linewidth=4, alpha=0.7)
      ax[i].set_title(titles[iy],fontsize=12, fontweight=1)
      ax[i].set_ylabel(ylabel)
      if iy == len(ys) - 1:
        ax[i].set_xlabel(xlabel)
      iy += 1
  plt.show()

def stacked_bar(ys,color,legend):
  plt.style.use('seaborn-darkgrid')

  plt.bar(legend, ys, color =color, width = 0.4)
  plt.xlabel("Configuration used")
  plt.ylabel("r2 score")
  plt.title("r2 score per config file")
  plt.show()