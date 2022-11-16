import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import optuna
import xgboost as xgb
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import scipy.stats as st
import plotly.graph_objects as go
import missingno as msno
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,OneClassSVM
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import catboost as cb


from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score, classification_report,mean_squared_error
from scipy import stats

def get_models():
    models = list()
    models.append(BaggingRegressor(n_estimators=100))
    models.append(RandomForestRegressor(bootstrap=False, max_features='sqrt', min_samples_split=3,n_estimators=1000, random_state=42))
    models.append(ExtraTreesRegressor(max_features='log2', n_estimators=1000, random_state=2))
    models.append(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5))
    models.append(xgb.XGBRegressor(n_estimators=100))
    models.append(CatBoostRegressor(loss_function='RMSE'))
    models.append(MLPRegressor())
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(OneClassSVM())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(KNeighborsRegressor())
    models.append(AdaBoostRegressor())
    return models
'''
def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "criterion": trial.suggest_categorical("criterion", ['squared_error']),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)
    }

    rf = ExtraTreesRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy
def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "criterion": trial.suggest_categorical("criterion", ['squared_error']),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)
    }

    rf = RandomForestRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy
def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
    }

    rf = BaggingRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy
def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "n_estimators": trial.suggest_int("n_estimatorsGBR", 100, 1000),
        "learning_rate": trial.suggest_categorical('learning_rateGBR',[0.03, 0.3,0.1]),
        "max_depth": trial.suggest_int("max_depthGBR", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_splitGBR", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leafGBR", 2, 10)
    }

    rf = GradientBoostingRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy

def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "learning_rate": trial.suggest_categorical('learning_rateXGB', [0.03, 0.3, 0.1]),  # default 0.1
        "max_depth": trial.suggest_int('max_depthXGB', 2, 20),  # default 3
        "n_estimators": trial.suggest_int('n_estimatorsXGB', 100, 1000),  # default 100
        "subsample": trial.suggest_categorical('subsample', [0.6, 0.4])
    }

    rf = xgb.XGBRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy


def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)
    }

    rf = DecisionTreeRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy

def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "n_neighbors": trial.suggest_categorical("n_neighbors", [5,6,7,8,9,10]),
        "leaf_size": trial.suggest_int("leaf_size", 30, 300),
    }

    rf = KNeighborsRegressor(**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy
'''
def objective(trial):
    data, target = scaled_df,labels
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "loss": trial.suggest_categorical("loss", ['linear', 'square', 'exponential']),
    }

    rf = AdaBoostRegressor(base_estimator=ExtraTreesRegressor(),**param)

    rf.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = rf.score(valid_x, valid_y)
    return accuracy



X = pd.read_csv('train1.csv',sep=';')
X_findAns = pd.read_csv('test1.csv',sep=';')
Id = X_findAns["id"]
tmp = X.columns

correlation = X.corr()
print(correlation['quality'].sort_values(ascending = False),'\n')

X = X.drop(['density'],axis=1)
X_findAns = X_findAns.drop(['density'],axis=1)



features, labels,types=  X.drop(["quality","type"],axis=1),X["quality"],X["type"]
features1,types1 = X_findAns.drop(["id","type"],axis=1),X_findAns["type"]
scaler = preprocessing.StandardScaler().fit(features1)
names = features.columns


d = scaler.transform(features)
d1 = scaler.transform(features1)

scaled_df = pd.DataFrame(d, columns=names)
scaled_df1 = pd.DataFrame(d1, columns=names)
scaled_df = pd.concat([scaled_df,types],axis=1)
scaled_df1 = pd.concat([scaled_df1,types1],axis=1)

X_train, X_test, y_train, y_test=train_test_split(scaled_df,labels,test_size=0.05,random_state=42)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=80, timeout=900)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



'''
fname3 = 'submitFinal.csv'
f=open(fname3,'w')
for i in range (-1,len(ans)):
    if (i==-1):
        f.write('id,quality')
    else:
        f.write( str(round(Id.iloc[i])) + ',' + str(ans[i]) )
    f.write('\n')
f.close()
'''