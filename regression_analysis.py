import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn import metrics


with open("data/fulldataset.json") as f:
    df = pd.read_json(json.load(f))
"""
columns =
['points', 'price', 'variety', 'region_2', 'vintage', 'winery_pare',
       'COUNTY', 'slope_l', 'slope_r', 'slope_h', 'tfact', 'wei', 'elev_l',
       'elev_r', 'elev_h', 'nirrcapcl', 'otherph', 'weg', 'drainagecl',
       'nirrcapscl', 'hydgrp', 'taxorder', 'taxsuborder', 'taxgrtgroup',
       'taxsubgrp', 'taxpartsize', 'taxtempcl', '_-3_T', '_-2_T', '_-1_T',
       '_0_T', '_1_T', '_2_T', '_3_T', '_4_T', '_5_T', '_6_T', '_7_T', '_8_T',
       '_9_T', '_-3_P', '_-2_P', '_-1_P', '_0_P', '_1_P', '_2_P', '_3_P',
       '_4_P', '_5_P', '_6_P', '_7_P', '_8_P', '_9_P']
"""
df.drop(['vintage','COUNTY','price','region_2'],axis=1,inplace=True)
df["otherph"] = df["otherph"].str.replace(">","g")
df["otherph"] = df["otherph"].str.replace("<","l")
df = pd.get_dummies(df)
print(list(df.columns))
df.dropna(axis=0,how="any",inplace=True)
df.reset_index(drop=True, inplace=True)

X,Y = df.loc[:,df.columns != "points"],df["points"]
print(X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
SVR_pipe = Pipeline([('scaler',StandardScaler()),('clf',LinearSVR(C=1,epsilon=0.0,max_iter=5000,\
                    dual=False,random_state=42,loss='squared_epsilon_insensitive'))])
#SVR_pipe = Pipeline([('scaler',StandardScaler()),('clf',LinearSVR(C=1,dual=False))])

SVR_pipe.fit(X_train,Y_train)

for i,j in sorted([(i,np.absolute(j)) for i,j in zip(X_train.columns,SVR_pipe.named_steps['clf'].coef_)], key = lambda x: x[1])[::-1]:
    print(i,",",j)

score = SVR_pipe.score(X_train,Y_train)
print(score)
cv_score = cross_val_score(SVR_pipe,X_train,Y_train, cv=10)
print(cv_score.mean())
Y_pred = SVR_pipe.predict(X_test)
print(metrics.r2_score(Y_test,Y_pred))
print(metrics.mean_absolute_error(Y_test,Y_pred))
plt.scatter(Y_test,Y_pred,alpha=0.15)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.show()
