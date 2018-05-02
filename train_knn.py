# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 21:34:49 2018

@author: HGiga2
"""
#C:\Users\HGiga2\Anaconda3\Lib\site-packages\sklearn\datasets\data
import numpy as np
import pandas as pd
from sklearn import datasets
# 引入 train_test_split 分割方法，注意在 sklearn v0.18 後 train_test_split 從 sklearn.cross_validation 子模組搬到 sklearn.model_selection 中
from sklearn.model_selection import train_test_split
# 引入 KNeighbors 模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import pickle 

# 引入 iris 資料集
raw_iris = datasets.load_iris()
# 探索性分析 Exploratory data analysis，了解資料集內容

#df_X = pd.DataFrame(raw_iris.data)
df_X = raw_iris.data
print("start")
print(df_X )
print("end")
# target 為預測變數
df_y = pd.DataFrame(raw_iris.target)
#df_y = iris.target
print(df_y)
print("end y")

df_y=pd.DataFrame(raw_iris.target)

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.005)
print(len(df_y))
# 印出切分 y_train 的數量為所有資料集的 70%，共 105 筆
print(y_train)
print(len(y_train))
# 印出切分的 y_test 資料為所有資料集的 30%，共 45 筆
print(y_test)
#print(len(y_test))
# 初始化 LinearSVC 實例
lin_clf = LinearSVC()
# 使用 fit 來建置模型，其參數接收 training data matrix, testing data array，所以進行 y_train.values.ravel() Data Frame 轉換
lin_clf.fit(X_train, y_train.values.ravel())
# 初始化 KNeighborsClassifier 實例
knn = KNeighborsClassifier()
# 使用 fit 來建置模型，其參數接收 training data matrix, testing data array，所以進行 y_train.values.ravel() 轉換
knn.fit(X_train, y_train.values.ravel())
model=knn.fit(X_train, y_train.values.ravel())




# 使用 X_test 來預測結果
print(lin_clf.predict(X_test))
lin_result=lin_clf.predict(X_test)
print(X_test[0])
print(y_test)

#print(y_test[1])
print(lin_result[0])

print(lin_clf.score(X_test, y_test))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'), 2)


# 使用 X_test 來預測結果
print(knn.predict(X_test))
g=knn.predict_proba(X_test)
# 印出 testing data 預測標籤機率
print(knn.predict_proba(X_test))

# 印出預測準確率
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
    #  print(num)
    #  print("x")
     # print(x[j])
     # print("y")
      #print(y[j])
      #print("z")
      #print(z[j])
      #print("------")
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


    
  

