import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('heart.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = df.copy()
#input columns
X = data.iloc[:,0:13]  
#target column 
y = data.iloc[:,-1]    
#apply SelectKBest class to extract top best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(12,'Score'))  #print best features
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap="magma",fmt='.2f')
for i in df.columns:
    print(i,len(df[i].unique()))
df.drop("slope",axis=1,inplace=True)    
df.drop("thal",axis=1,inplace=True)
df.drop("restecg",axis=1,inplace=True)
df.drop("trestbps",axis=1,inplace=True)
df.head()
x = df.iloc[:,0:9] # Features
y = df.iloc[:,9] # Target variable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 10)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
from sklearn import metrics
er=metrics.accuracy_score(y_pred,y_test)
f1=metrics.f1_score(y_pred,y_test)
print('f1 Score: ',f1)
print('precision: ',metrics.precision_score(y_pred,y_test))
print('recall: ',metrics.recall_score(y_pred,y_test))
print('Accuracy Score: ',er)
metrics.plot_confusion_matrix(classifier,X_test,y_test,display_labels=['0','1'])
df2=pd.DataFrame(y_pred,columns=['predicted_values'])
df2.head()
test_arr=np.array(y_test)
test_arr
df3=pd.DataFrame(test_arr,columns=['Actual_values'])
df3.head()
df2=pd.concat([df2,df3],axis=1)
df2.head()
print("correct predictions:")
for j in range(len(df2)):
  if(df2['predicted_values'].loc[j]==df2['Actual_values'].loc[j]):
    print(df2['predicted_values'].loc[j]," ",df2['Actual_values'].loc[j]," ",j)
print("Incorrect predictions:")
for j in range(len(df2)):
  if(df2['predicted_values'].loc[j]!=df2['Actual_values'].loc[j]):
    print(df2['predicted_values'].loc[j]," ",df2['Actual_values'].loc[j]," ",j) 
    
import pickle
pickle.dump(classifier,open('model.pkl','wb'))
pickle.dump(sc,open('sc.pkl','wb'))       