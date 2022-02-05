import random 
import enum
import numpy as np
import pandas as pd

datset = "/content/drive/MyDrive/Pattern Lab/Datsets/blobs.txt"

data = np.genfromtxt(datset, skip_header=True)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import colorsys

def plot_input_data(X, data_title="train"):
    X = np.array(X, copy=False)
    plt.title(f'Blobs {data_title.capitalize()} Data') 
    plt.xlabel('F1') 
    plt.ylabel('F2') 
    plt.xticks() 
    plt.yticks() 
    plt.scatter(*X.T, s=20)
    plt.grid(True, which='both')
    plt.show()

plot_input_data(data, data_title="whole")

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(data,
                                   test_size=0.2,
                                   random_state=911)

plot_input_data(x_train, data_title="train")

plot_input_data(x_test, data_title="test")

class Metric(enum.Enum):
  EUCLIDEAN_DISTANCE = "euclidean"
  MANHATTAN_DISTANCE = "manhattan"
  COSINE_SIMILARITY = "cosine"

from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
class K_Means:
  def fit(self, features, K, metric, max_iter):
    self.metric = metric
    random_index = random.sample(range(len(features)), K)
    means = []
    for i in random_index:
      means.append(list(features[i]))
    for iter in range(max_iter):
      classes = []
      for p in features:
        distance=[]
        for m in means:
          distance.append(self.compute_distance(m, p))
        if self.metric==Metric.COSINE_SIMILARITY:
          classes.append(np.argmax(distance))
        else:
          classes.append(np.argmin(distance))
      new_means = []
      for i in range(len(means)):
        x_axis = []
        y_axis = []
        for index, item in enumerate(classes):
          if i == item:
            x_axis.append(features[index][0])
            y_axis.append(features[index][1])
        x = np.mean(x_axis)
        y = np.mean(y_axis)
        new_means.append([x, y])
      if means == new_means:
        break
      means = new_means
    return means  

  def predict(self, features, means, K):
    predictions = []
    for p in features:
        distance=[]
        for m in means:
          distance.append(self.compute_distance(m, p))
        if self.metric==Metric.COSINE_SIMILARITY:
          predictions.append(np.argmax(distance))
        else:
          predictions.append(np.argmin(distance))
    return predictions

  def compute_distance(self, x1, x2):
    if self.metric == Metric.EUCLIDEAN_DISTANCE:
      dist = distance.euclidean(x1, x2)
    elif self.metric == Metric.MANHATTAN_DISTANCE:
      dist = manhattan_distances([x1], [x2])
    elif self.metric == Metric.COSINE_SIMILARITY:
      dist = cosine_similarity([x1], [x2])
    return dist

def plot_data_after_KMeans(X, means, cluster_assignments, K):
    X = np.array(X, copy=False)
    means = np.array(means, copy=False)
    cluster_assignments = np.array(cluster_assignments, copy=False)

    plt.title('Blobs Data Clusters' + '\n K='+str(K)) 
    plt.xlabel('F1') 
    plt.ylabel('F2') 
    plt.xticks() 
    plt.yticks()
  
    colors = np.array([color['color'] for color in plt.rcParams["axes.prop_cycle"]])
    if len(colors) < K:
        plt.scatter(*X.T, c = cluster_assignments, s=20)
    else:
        plt.scatter(*X.T, c = colors[cluster_assignments], s=20)
    plt.scatter(*means.T, marker='2', c="black", s=100)
    plt.grid(True, which='both')
    plt.show()

K_samples = [2, 3, 4]
for k in K_samples:
    kmeans = K_Means()
    means = kmeans.fit(x_train, K=k, metric=Metric.EUCLIDEAN_DISTANCE, max_iter=500)
    train_predict = kmeans.predict(x_train, means, K=k)
    plot_data_after_KMeans(x_train, means=means, 
                           cluster_assignments=train_predict, K=k)
    print()
    test_predict = kmeans.predict(x_test, means, K=k)
    plot_data_after_KMeans(x_test, means=means, 
                           cluster_assignments=test_predict, K=k)
    print()

K_samples = [2, 3, 4]
for k in K_samples:
    kmeans = K_Means()
    means = kmeans.fit(x_train, K=k, metric=Metric.COSINE_SIMILARITY, max_iter=500)
    train_predict = kmeans.predict(x_train, means, K=k)
    plot_data_after_KMeans(x_train, means=means, 
                           cluster_assignments=train_predict, K=k)
    print()
    test_predict = kmeans.predict(x_test, means, K=k)
    plot_data_after_KMeans(x_test, means=means, 
                           cluster_assignments=test_predict, K=k)
    print()