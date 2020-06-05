import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

#reading the data
data = pd.read_csv('diabetes.csv')

data.info()
data.isnull().sum()

corr = data.corr(method = 'pearson')
sns.heatmap(corr,)


X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, -1].values

#splitting the dataset into training  (80%)  and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


#computing euclidean distance
def ECD(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
       
        #computing the euclidean distance of all value of x
        distances = [ECD(x, x_train) for x_train in self.X_train]
        
        # sorting the destance and selecting the k neighbors
        k_idx = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

#findeg the best value of k
#score_list =[]
# for i in range(15):
#     classifier=KNN(i)
#     classifier.fit(X_train,y_train)
#     y_pred=classifier.predict(X_test) 
#     print(accuracy_score(y_test, y_pred))
# print(score_list)
# print(max(score_list))



#for k = 10 we getting good score
classifier=KNN(10)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test) 

#comfusion martix for finding the accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))


TN=cm[0][0]
FN=cm[1][0]
TP=cm[1][1]
FP=cm[0][1]
accuracy=(TN+TP)/(TN+TP+FN+FP)
precision=(TP)/(TP+FP)
recall=TP/(TP+FN)

print('Accuracy: ',accuracy,'\nPrecision:',precision,'\nRecall:',recall)



