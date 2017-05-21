import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = "/User/xxxxxx"
df = pd.read_csv(file, header=None)
df.head(10)

y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

x = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker="o",label="setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker="x", label="versicolor")
plt.xlabel("花瓣长度")
plt.ylabel("花径长度")
plt.legend(loc='upper left')
plt.show()