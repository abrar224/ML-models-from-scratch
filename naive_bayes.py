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

data_x = df.loc[:, df.columns != "Churn"]
data_y = df["Churn"]

import random
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    train_size = 0.8,
                                                    stratify = data_y,
                                                    random_state = 911)

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

categorical_columns = x_train.columns.drop(numeric_columns)

class Naive_Bayes:
    
  def fit(self, features, labels):
    self.classes = np.unique(labels)
    self.prior = (features.groupby([labels]).apply(lambda x: len(x))/
                      features.shape[0]).to_numpy() 
    self.likelyhood = []
    for col in features.columns:
      if col in categorical_columns:
        label_count = features.groupby([labels]).size().to_numpy() 
        likelyhoods = features[col].groupby([labels, features[col]]).size().to_dict()
        for keys in likelyhoods.keys():
          if 0 in keys:
            likelyhoods[keys] = likelyhoods[keys] / label_count[0]            
          elif 1 in keys:
            likelyhoods[keys] = likelyhoods[keys] / label_count[1]
        self.likelyhood.append(likelyhoods)

      elif col in numeric_columns:
        self.mean = features[col].groupby(labels).apply(np.mean).to_numpy()
        self.var = features[col].groupby(labels).apply(np.var).to_numpy()

  def predict(self, features):
    y_pred = []
    for row in features[categorical_columns].to_numpy():
      posterior = []
      for index in self.classes:
        priors = self.prior[index]
        likelyhoods = 1
        for i, item in enumerate(row):
          likelyhoods *= self.likelyhood[i][index, item]
        posteriors = priors * likelyhoods
        posterior.append(posteriors)
      y_pred.append(self.classes[np.argmax(posterior)])
    return y_pred

  def predict_(self, features):
    y_pred = []    
    for x in features[numeric_columns].to_numpy():
      posterior = []
      for index, i in enumerate(self.classes):
        priors = np.log(self.prior[index])
        condition = np.sum(np.log(self.pdf(index, x)))
        posteriors = priors + condition
        posterior.append(posteriors)
      y_pred.append(self.classes[np.argmax(posterior)])
    return y_pred

  def pdf(self, class_index, x):
    mean = self.mean[class_index]
    var = self.var[class_index]
    n = np.exp(-((x-mean)**2 / (2*var)))
    m = np.sqrt(2* np.pi * var)
    probability = n / m
    return probability

# with stratification 
nb = Naive_Bayes()
nb.fit(x_train, y_train)
pred = nb.predict(x_test)
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

# without stratification
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                    train_size = 0.8,
                                                    random_state = 911)
nb = Naive_Bayes()
nb.fit(x_train, y_train)
pred = nb.predict(x_test)
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