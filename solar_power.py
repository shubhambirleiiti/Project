import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv (r"C:\Users\projectdataset\spg.csv")
print(df)
print(df.info())
print(df.columns )
#corr=df.corr()

print(df.info())
corr=df.corr()
sns.heatmap (corr,annot=True)
plt.show()
X=df.iloc[:,:21]
y=df.iloc[:,-1]
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= 0.8)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
y_test =pd.DataFrame (y_test )
y_test.reset_index(inplace=True)

y_test.drop("index",inplace= True,axis=1)
print(y_test )
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
lin=LinearRegression()
lin.fit(X_train ,y_train )
lin_pred=lin.predict(X_test)
lin_pred=pd.DataFrame (lin_pred )
lin_pred.columns=["lin_pred"]
bag=BaggingRegressor ()
boost=AdaBoostRegressor()
Xboost=GradientBoostingRegressor ()
bag.fit(X_train ,y_train )
bag_pred=bag.predict(X_test)
bag_pred=pd.DataFrame (bag_pred )
bag_pred.columns=["bag_pred"]
boost .fit(X_train ,y_train )
boost_pred=boost.predict(X_test)
boost_pred=pd.DataFrame (boost_pred )
boost_pred.columns=["boost_pred"]
Xboost .fit(X_train ,y_train )
Xboost_pred=Xboost .predict(X_test)
Xboost_pred=pd.DataFrame (Xboost_pred )
Xboost_pred.columns=["Xboost_pred"]
y_result=pd.concat([y_test,lin_pred,bag_pred,boost_pred,Xboost_pred],axis=1)
print(y_result)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
rmse=cross_val_score(bag,X,y,scoring="neg_root_mean_squared_error")
print(rmse)
r2_lin=r2_score(y_test,lin_pred )
print(r2_lin )
r2_bag=r2_score(y_test,bag_pred )
print(r2_bag )
r2_boost=r2_score(y_test,boost_pred )
print(r2_boost)
r2_Xboost=r2_score(y_test,Xboost_pred)
print(r2_Xboost )

