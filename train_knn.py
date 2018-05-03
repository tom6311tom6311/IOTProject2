# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import pickle 

raw_iris = datasets.load_iris()
df_X = raw_iris.data
df_y = pd.DataFrame(raw_iris.target)
df_y=pd.DataFrame(raw_iris.target)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.005)
print(X_train.shape)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train.values.ravel())
model=knn.fit(X_train, y_train.values.ravel())
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'), 2)


g=knn.predict_proba(X_test)

print(knn.score(X_test, y_test))
x=[16,230,352,474,596,702,702,596,474,352,
   230,108,2,124,246,368,490,612,734,856,
   856,135,204,326,448,631,753,875,875,753,
   753,631,570,110,232,354,476,598,781,781,
   781,781,781,781,659,537,415,293,171,49,
   171,171,171,171,293,415,415,415,415,537,
   537,537,537,659,659,659,537,135,257,257,
   257,257,257,562,562,562,562,562,745,745,
   745,745,745,255,15,493 
   ]
y=[801,801,801,801,801,801,684,684,684,684,
   684,684,684,440,440,440,440,440,440,440,
   196,34,34,34,34,34,34,34,156,156,
   178,156,278,763,763,763,763,763,702,580,
   458,336,214,92,92,92,92,92,92,92,
   214,336,458,580,580,519,397,275,214,214,
   336,458,580,519,397,275,275,40,40,223,
   406,589,772,772,589,406,223,40,40,223,
   406,589,711,820,820,820
   ]
z=[63,63,63,63,63,63,63,63,63,63,
   63,63,63,63,63,63,63,63,63,63,
   63,63,63,63,63,63,63,63,63,63,
   63,63,63,20,20,20,20,20,20,20,
   20,20,20,20,20,20,20,20,20,20,
   20,20,20,20,20,22,22,22,22,20,
   20,20,20,20,20,20,20,188,188,188,
   188,188,188,188,188,188,188,188,188,188,
   188,188,188,134,134,134]
j=0
res=0
resy=0
resz=0
for num in g[0]:
   
    if num!= 0:
      resy=num*y[j]+resy
      res=num*x[j]+res    
      resz=num*z[j]+resz
    j=j+1
print("final ")
print("x=")
print(res)
print("y=")
print(resy)
print("z=")
print(resz)
