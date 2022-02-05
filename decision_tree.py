import numpy as np
import pandas as pd

df = pd.read_csv("Telco-Customer-Churn.csv", na_values=" ")

df.dropna(inplace=True)
df.reset_index(drop=True)

df = df.replace ({
    "Churn":{
        "No": 0,
        "Yes": 1
    }
})

df.drop(labels=["customerID", "gender", "SeniorCitizen", "Partner",
                "Dependents"],
        axis=1,
        inplace=True)

numeric_columns=["tenure", "MonthlyCharges", "TotalCharges"]
df.drop(labels=numeric_columns, axis=1, inplace=True)

categorical_columns = df.columns

data_x = df.to_numpy()
data_y = df["Churn"]

import random
import math
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    train_size = 0.8,
                                                    stratify = data_y,
                                                    random_state = 911)

class Node:
    def __init__(self, attribute=None, attribute_values=None, child_nodes=None, decision=None):
        self.attribute = attribute
        self.attribute_values = attribute_values
        self.child_nodes = child_nodes
        self.decision = decision


class DecisionTree:
    root = None
    @staticmethod
    def plurality_values(data):
        labels = data[:, data.shape[1] - 1] # store the last column in labels
        if np.count_nonzero(labels==0) > np.count_nonzero(labels==1): 
          return 0
        else: 
          return 1

    @staticmethod
    def all_zero(data):
        labels = data[:, data.shape[1] - 1]
        return np.count_nonzero(labels==0) == len(labels)

    @staticmethod
    def all_one(data):
        labels = data[:, data.shape[1] - 1] 
        return np.count_nonzero(labels==1) == len(labels)
        
        
    @staticmethod
    def importance(data, attributes):
        labels = data[:, data.shape[1] - 1] 
        p = np.count_nonzero(labels==1)
        n = np.count_nonzero(labels==0)
        q = p/(p+n)
        entropy_parent=-q*math.log(q,2)-(1-q)*math.log(1-q,2)        
        gain = {}
        for i in attributes:
          entropy_childs = 0
          features = np.unique(data[:, i])
          for x in features:
            f = set(np.where(data[:, i] == x)[0])
            class_0 = np.where(labels == 0)[0]
            class_1 = np.where(labels == 1)[0]
            nk = len(f.intersection(class_0))
            pk = len(f.intersection(class_1))
            qk = pk/(pk+nk)
            try:
              entropy_child=-qk*math.log(qk,2)-(1-q)*math.log(1-qk,2)
            except:
              entropy_child = 0
            entropy_childs+=((pk+nk)/(p+n))*entropy_child 
          gain[i] = entropy_parent - entropy_childs
        for key, value in gain.items():
          if value == max(gain.values()):
            return key 

    def train(self, data, attributes, parent_data):
        data = np.array(data)
        parent_data = np.array(parent_data)
        attributes = list(attributes)
        if data.shape[0] == 0:  # if x is empty
            return Node(decision=self.plurality_values(parent_data))
        elif self.all_zero(data):
            return Node(decision=0)
        elif self.all_one(data):
            return Node(decision=1)
        elif len(attributes) == 0:
            return Node(decision=self.plurality_values(data))
        else:
            a = self.importance(data, attributes)
            tree = Node(attribute=a, attribute_values=np.unique(data[:, a]), child_nodes=[])
            attributes.remove(a)
            for vk in np.unique(data[:, a]):
                new_data = data[data[:, a] == vk, :]
                subtree = self.train(new_data, attributes, data)
                tree.child_nodes.append(subtree)
            return tree

    def fit(self, data):
        self.root = self.train(data, list(range(data.shape[1] - 1)), np.array([]))

    def predict(self, data):
        predictions = []
        for i in range(data.shape[0]):
            current_node = self.root
            while True:
                if current_node.decision is None:
                    current_attribute = current_node.attribute
                    current_attribute_value = data[i, current_attribute]
                    if current_attribute_value not in current_node.attribute_values:
                        predictions.append(random.randint(0, 1))
                        break
                    idx = list(current_node.attribute_values).index(current_attribute_value)
                    current_node = current_node.child_nodes[idx]
                else:
                    predictions.append(current_node.decision)
                    break
        return predictions

clf = DecisionTree()
clf.fit(x_train)
pred = clf.predict(x_test)
accuracy = np.sum(y_test == pred)/len(y_test)
print("Accuracy:",accuracy)
TP = 0
TN = 0
FN = 0
FP = 0
for y1 ,y2 in zip(y_test, pred):
  if y1==1 and y2==1:
    TP += 1
  elif y1==0 and y2==0:
    TN += 1
  elif y1==1 and y2==0:
    FN += 1
  elif y1==0 and y2==1:
    FP += 1
pr = TP / (TP + FP)
rl = TP / (TP + FN)
f1 = 2 * (pr * rl) / (pr + rl)
print("Precision:", pr)
print("Recall:", rl)
print("F1 score:", f1)