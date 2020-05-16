
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix




#converting the data to .csv file forment with column names

# read_file = pd.read_csv (r'spambase.data', header = None)
# read_file.columns = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove',
# 'word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses',
# 'word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000',
# 'word_freq_money','word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet',
# 'word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology',
# 'word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
# 'word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#',
# 'capital_run_length_average','capital_run_length_longest','capital_run_length_total','spam_or_not']
# read_file.to_csv (r'spam.csv')


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


#smv with linear kaernal
# C = 1000 wr are getting 82% accuracy
svmlclf = svm.LinearSVC(C=1000)
svmlclf.fit(X_train, y_train)
y_pred = svmlclf.predict(X_test)
pa = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(pa)
from sklearn.metrics import confusion_matrix
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