##flight price prediction
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train_df=pd.read_csv(r"C:\Data_Train.csv")
print(train_df.head())
print(train_df .info())
print(train_df.shape)
test_df=pd.read_csv (r"C:\Test_set.csv")
print(test_df .head())
print(test_df .info())
print(test_df.shape)
final_df=train_df .append(test_df)
print(final_df .head())
print(final_df .info())
print(final_df.shape)
print(final_df.describe() )
#feature engineering  start
final_df["Date_of_Journey"]=pd.to_datetime(final_df ["Date_of_Journey"])
print(final_df .info())
final_df["Day"]=final_df ["Date_of_Journey"].dt.day
final_df["Month"]=final_df ["Date_of_Journey"].dt.month
final_df["Year"]=final_df ["Date_of_Journey"].dt.year
print(final_df.head().to_string())
final_df.drop("Date_of_Journey",axis=1,inplace=True)
print(final_df .head() )
print(final_df.corr())
fd=final_df.corr()
#sns.heatmap (fd,annot=True,cmap="viridis")
#plt.colorbar()
#plt.show()
final_df["Day"]=final_df ["Day"].astype(float)
print(final_df.info())
#final_df.drop ("Duration",inplace=True,axis=1)
#print(final_df["Arrival_Time"].str.split(' '))
#print(final_df["Arrival_Time"] )
#print(final_df.isnull ().sum())
#final_df["Arrival_hour"] =final_df["Arrival_Time"].str.split(":").str[0]
#final_df["Arrival_min"] =final_df["Arrival_Time"].str.split(":").str[1]
#final_df["Arrival_hour"] =final_df["Arrival_hour"].astype(int)
#final_df["Arrival_min"] =final_df["Arrival_min"].astype(int)
final_df.drop("Arrival_Time",axis=1,inplace=True)
final_df["Dep_hour"] =final_df["Dep_Time"].str.split(":").str[0]
final_df["Dep_min"] =final_df["Dep_Time"].str.split(":").str[1]
final_df["Dep_min"] =final_df["Dep_min"].astype(int)
final_df["Dep_hour"] =final_df["Dep_hour"].astype(int)
final_df.drop("Dep_Time",axis=1,inplace=True)
print(final_df .info())
final_df["Total_Stops"]=final_df ["Total_Stops"].map({"non-stop":0,"2 stops":2,"1 stop":1,"3 stops":3,"4 stops":4})
final_df[final_df["Total_Stops"].isnull()]
print(final_df.head().to_string())
#print(final_df.info())
#final_df.drop("Arrival_min",inplace= True,axis=1)
final_df.drop("Dep_min",inplace= True,axis=1)
final_df.drop("Route",inplace=True,axis=1)
final_df["Duration_hour"]=final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]
print(final_df[final_df ["Duration_hour"]=="5m"])
final_df.drop(6474,axis=0,inplace=True)
final_df.drop(2660,axis=0,inplace=True)
print(final_df[final_df ["Duration_hour"]=="5m"])
final_df["Duration_hour"]=final_df["Duration_hour"].astype(int)
final_df.drop ("Duration",inplace=True,axis=1)
print(final_df.info())
print(final_df["Airline"].unique())
from sklearn .preprocessing import LabelEncoder
labelencoder=LabelEncoder ()
final_df ["Airline"]=labelencoder .fit_transform(final_df ["Airline"])
final_df ["Source"]=labelencoder .fit_transform(final_df ["Source"])
final_df ["Destination"]=labelencoder .fit_transform(final_df ["Destination"])
final_df ["Additional_Info"]=labelencoder .fit_transform(final_df ["Additional_Info"])
print(final_df.info())
pd.get_dummies(final_df,columns=["Airline","Source","Destination","Additional_Info"],drop_first=True)
#test_data.drop("Price",inplace= True,axis=1)
#print(final_df.to_string() )
print(final_df.shape)
final_df.dropna(inplace=True)
test_data=final_df[final_df["Price"].isnull()]
train_data=final_df [~final_df ["Price"].isnull()]
print(train_data .info())
print(train_data .shape)
print(train_data )
#final_df.dropna(inplace=True)
X=train_data[ ["Airline","Source","Destination","Total_Stops","Additional_Info","Day","Month","Year","Dep_hour"]]
y=train_data [["Price"]]
X_test1=test_data[ ["Airline","Source","Destination","Total_Stops","Additional_Info","Day","Month","Year","Dep_hour"]]
#print(final_df[final_df=="Nan"])
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= .8)
y_test.reset_index(inplace=True)
print(y_test)
y_test.drop("index",inplace=True,axis=1)
from sklearn.ensemble import RandomForestRegressor
random_regr=RandomForestRegressor ()
random_regr.fit(X_train ,y_train )
y_predicted=random_regr .predict(X_test)
y_pred=pd.DataFrame (y_predicted )
#y_pred.columns.rename({0:"Predicted"})
y_pred.columns =["predicted_RF"]
print(y_pred.head())
print(y_test.head())
from sklearn.metrics import r2_score
r2=r2_score (y_test,y_pred)
print("r2_RF=",r2)
print(X_train .shape)
print(X_test .shape)
print(y_train .shape)
print(y_test .shape)
print(X_train .head())
train_data_reshaped=X_train.values.reshape(8544,9,1)
train_label_reshaped=y_train.values.reshape(8544,1,1)
test_data_reshaped=X_test.values.reshape(2136,9,1)
test_label_reshaped=y_test.values.reshape(2136,1,1)
n_timesteps=train_data_reshaped.shape[1]
n_features=train_data_reshaped .shape[2]
print(test_data_reshaped[0].shape)
import tensorflow as tf
from tensorflow import keras
from keras import Sequential ,Model ,layers
model=Sequential()
model.add(keras.layers.Input(shape=(n_timesteps ,n_features )))
model.add(keras.layers.LSTM(150,return_sequences=True, activation="relu"))
model.add(keras.layers.LSTM (150,return_sequences=True,activation="relu"))
model.add(keras.layers.LSTM (150,return_sequences=True,activation="relu"))
model.add(keras.layers.LSTM (150,return_sequences=False,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
#model.add(Flatten())
model.add(keras.layers.Dense(1,activation="linear"))
model.compile(
    optimizer="Adam",
    loss="mean_squared_error",
    metrics=['accuracy'])
print(model.summary() )
#history=model.fit(train_data_reshaped ,train_label_reshaped ,epochs= 10)
#print(history)

model.fit(train_data_reshaped,train_label_reshaped ,epochs= 100)

#result = model.predict(padded_docs_test, verbose=2)
#result = result > threshold
y_pred1_nn=model.predict (test_data_reshaped  ).flatten()
print(y_pred1_nn)
y_pred1=pd.DataFrame (y_pred1_nn )
y_pred1.columns =["predicted_LSTM"]
print(y_pred1.head())
print(y_test.head())
y_result=pd.concat([y_test,y_pred,y_pred1],axis=1)
print(y_result)
from sklearn .metrics import r2_score
r2score=r2_score(y_test,y_pred1 )
print("r2 score=",r2score )
"""y_pred1["predicted"]=y_pred1["predicted"].map({True:1,False:0})
print(y_pred1 )
print(y_pred1["predicted"].value_counts() )"""

