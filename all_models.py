from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm

def random_forest(data,labels,max_depth,random_state):
    model = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(data, labels)
    return model

def my_xgboost(data,labels,max_depth,random_state):
    model = XGBRegressor()
    model.fit(data, labels)
    return model

def LinearRegression_model(data,labels,max_depth,random_state):
    model = LinearRegression()
    model.fit(data, labels)
    return model

def SVM_model(data,labels,max_depth,random_state):
    model = svm.SVC()
    model.fit(data, labels)
    return model



