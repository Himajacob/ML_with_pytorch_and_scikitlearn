import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None, encoding='utf-8')
print(df.head())

y = df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', 0, 1)

x = df.iloc[0:100, [0,2]].values

plt.scatter(x[:50,0], x[:50,1],color='red', marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()

class Perceptron:
    def __init__(self, lr=0.01, n_iter=50, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []

        for  _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x,y):
                update = self.lr*(target - self.predict(xi))
                self.w_ += update*xi
                self.b_+= update
                errors += int(update!=0.0)
            self.errors_.append(errors)
    
    def net_input(self, x):
        return np.dot(x,self.w_) + self.b_
    
    def predict(self, x):
        return np.where(self.net_input(x)>=0.0,1, 0)

ppn = Perceptron(lr=0.1, n_iter=10)
ppn.fit(x,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

def plot_decision_regions(x,y, classifier, resolution=0.02):
    markers =('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:,[0]].min() -1, x[:,[0]].max()+1
    x2_min, x2_max = x[:,[1]].min() -1, x[:,[1]].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(),xx2.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl,0], y=x[y ==cl,1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')

plot_decision_regions(x,y,classifier=ppn)
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()
