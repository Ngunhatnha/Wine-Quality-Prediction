import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import optuna
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import LinearSVR,OneClassSVM
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, mean_squared_error
from scipy import stats
from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import catboost as cb

def get_models():
    models = list()
    models.append(BaggingRegressor(n_estimators=178))
    models.append(RandomForestRegressor(max_depth=25,min_samples_leaf=2, min_samples_split=2,n_estimators=896, random_state=42))
    models.append(ExtraTreesRegressor(max_features='log2', n_estimators=312,max_depth=29,min_samples_leaf=2,min_samples_split=2, random_state=42))
    models.append(GradientBoostingRegressor(n_estimators=914, learning_rate=0.03,max_depth=11,min_samples_leaf=5, min_samples_split=9,random_state =42))
    models.append(xgb.XGBRegressor(n_estimators=489,learning_rate=0.03,max_depth=13,subsample=0.6))
    models.append(CatBoostRegressor(loss_function='RMSE'))
    models.append(MLPRegressor())
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(OneClassSVM())
    #models.append(SGDRegressor())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor(max_depth=15,min_samples_leaf=3,min_samples_split=4))
    models.append(KNeighborsRegressor(n_neighbors=7,leaf_size=240))
    models.append(AdaBoostRegressor(base_estimator=ExtraTreesRegressor(),n_estimators=54,loss='exponential'))
    return models


def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    # define split of data
    kfold = KFold(n_splits=9, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        count=0
        fold_yhats = list()
        # get data
        train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        meta_y.extend(test_y)
        # fit and make predictions with each sub-model
        for model in models:
            count+=1
            if(count==6):
                train_X1 = cb.Pool(train_X, train_y)
                test_X1 = cb.Pool(test_X, test_y)
                model.fit(train_X1)
                yhat = model.predict(test_X1)
                fold_yhats.append(yhat.reshape(len(yhat), 1))
            else:
                model.fit(train_X, train_y)
                yhat = model.predict(test_X)
                # store columns
                fold_yhats.append(yhat.reshape(len(yhat), 1))
        # store fold yhats as columns
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)


def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)


def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        mse = mean_squared_error(y, yhat)
        print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))


def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat), 1))
    meta_X = hstack(meta_X)
    # predict
    return meta_model.predict(meta_X)


def cap_outliers_IQR(df,column):
    upper_limit = df[column].mean() + 3*df[column].std()
    lower_limit = df[column].mean() - 3*df[column].std()
    df[column] = np.where(df[column] > upper_limit,upper_limit,df[column])
    df[column] = np.where(df[column] < lower_limit,lower_limit,df[column])
    return df[column]


X = pd.read_csv('train1.csv', sep=';')
X_findAns = pd.read_csv('test1.csv', sep=';')
Id = X_findAns["id"]
tmp = X.columns
#X=X.drop_duplicates()

correlation = X.corr()
print(correlation['quality'].sort_values(ascending=False), '\n')

X = X.drop(['density'],axis=1)
X_findAns = X_findAns.drop(['id','density'],axis=1)

'''
for i in range(0,len(X.columns)):
    b=cap_outliers_IQR(X,X.columns[i])
    if i==0:
        a=b
    else:
        a = pd.concat([a, b], axis=1)
X=a.dropna().reset_index()
X=X.drop(['index'],axis=1)
'''


features, labels, types = X.drop(["quality", "type"], axis=1), X["quality"], X["type"]
features1, types1 = X_findAns.drop(["type"], axis=1), X_findAns["type"]
scaler = preprocessing.StandardScaler().fit(features1)
names = features.columns

d = scaler.transform(features)
d1 = scaler.transform(features1)

scaled_df = pd.DataFrame(d, columns=names)
scaled_df1 = pd.DataFrame(d1, columns=names)
scaled_df = pd.concat([scaled_df, types], axis=1)
scaled_df1 = pd.concat([scaled_df1, types1], axis=1)

X_train, X_test, y_train, y_test = train_test_split(scaled_df, labels, test_size=0.05, random_state=42)

models = get_models()
# get out of fold predictions
meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
print('Meta ', meta_X.shape, meta_y.shape)
# fit base models
fit_base_models(X_train, y_train, models)
# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)
# evaluate base models
evaluate_models(X_test, y_test, models)
# evaluate meta model
yhat = super_learner_predictions(X_test, models, meta_model)
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_test, yhat))))

ans = super_learner_predictions(scaled_df1, models, meta_model)
print(ans)

fname3 = 'submitFinal_noVola.csv'
f = open(fname3, 'w')
for i in range(-1, len(ans)):
    if (i == -1):
        f.write('id,quality')
    else:
        f.write(str(round(Id.iloc[i])) + ',' + str(ans[i]))
    f.write('\n')
f.close()
