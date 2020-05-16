

# UMER#############################################

dataset = pd.read_csv('spam.csv')
dataset.info()
dataset.head()

#split the dataset into training data and testing data 
# X =  lables
# y = tagret variable 
X = dataset.iloc[:,0:57]
y = dataset.iloc[:,57:]



#splitting the dataset into training  (80%)  and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 1)

#applying the MLP SVM 
# with rbf kernel
# C = 1000 
# accuracy 90%
svmclf = svm.SVC(C=1000, kernel='rbf', gamma='scale',)
svmclf.fit(X_train, y_train)
y_pred = svmclf.predict(X_test)
#comparing the predicted value with actual value
y_test=y_test.to_numpy()
pa = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(pa)
#comfusion martix for finding the accuracy
cm = confusion_matrix( y_pred, y_test)
print("confusion matrix")
print(cm)


TN=cm[0][0]
FN=cm[1][0]
TP=cm[1][1]
FP=cm[0][1]
accuracy=(TN+TP)/(TN+TP+FN+FP)
precision=(TP)/(TP+FP)
recall=TP/(TP+FN)

print('Accuracy: ',accuracy,'\nPrecision:',precision,'\nRecall:',recall)



#  UMER##################################################




#svm with Quadratic kernel
# Polynomial Kernel of degree 2 is nothing but Quadratic kernel 
#C = 1000 wr are getting 77% accuracy
svmclf = svm.SVC(C=1000, kernel='poly', degree=2, gamma='scale',)
svmclf.fit(X_train, y_train)
y_pred = svmclf.predict(X_test)
#comparing the predicted value with actual value
# y_test=y_test.to_numpy()
pa = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(pa)
cm = confusion_matrix( y_pred, y_test)
print("confusion matrix")
print(cm)
TN=cm[0][0]
FN=cm[1][0]
TP=cm[1][1]
FP=cm[0][1]
accuracy=(TN+TP)/(TN+TP+FN+FP)
precision=(TP)/(TP+FP)
recall=TP/(TP+FN)
print('Accuracy: ',accuracy,'\nPrecision:',precision,'\nRecall:',recall)