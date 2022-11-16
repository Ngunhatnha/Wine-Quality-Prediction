import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import optuna
import xgboost as xgb
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score, classification_report,mean_squared_error
from scipy import stats

def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
	return scores

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

    paramLS = {
        "eps": trial.suggest_categorical("eps", [0.003,0.001,0.01]),
        "n_alphas": trial.suggest_categorical("n_alphas", [200,1000,500]),
        "max_iter": trial.suggest_categorical("max_iterLS", [3000])
    }

    paramSVR = {
        "epsilon": trial.suggest_categorical("epsilon", [0.01,0.001,0.003]),
        "max_iter": trial.suggest_categorical("max_iterSVR", [3000])
    }

    paramRFT = {
        "n_estimators": trial.suggest_int("n_estimatorsRFT", 100, 1000),
        "criterion": trial.suggest_categorical("criterionRFT", ['squared_error']),
        "max_depth": trial.suggest_int("max_depthRFT", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_splitRFT", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leafRFT", 2, 10)
    }

    paramGBR = {
        "n_estimators": trial.suggest_int("n_estimatorsGBR", 100, 1000),
        "learning_rate": trial.suggest_categorical('learning_rateGBR',[0.03, 0.3,0.1]),
        "max_depth": trial.suggest_int("max_depthGBR", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_splitGBR", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leafGBR", 2, 10)
    }

    paramXGB = {
        "learning_rate": trial.suggest_categorical('learning_rateXGB',[0.03, 0.3,0.1]),  # default 0.1
        "max_depth": trial.suggest_int('max_depthXGB',2, 20),  # default 3
        "n_estimators": trial.suggest_int('n_estimatorsXGB',100, 1000),  # default 100
        "subsample": trial.suggest_categorical('subsample',[0.6, 0.4])
    }

    estimators = [('ls', LassoCV(**paramLS)),
                  ('svr', LinearSVR(**paramSVR)),
                  ('rfr', RandomForestRegressor(**paramRFT)),
                  ('GBR', GradientBoostingRegressor(**paramGBR)),
                  ('Rid',RidgeCV())]
    tmep = StackingRegressor(estimators=estimators, final_estimator=xgb.XGBRegressor(**paramXGB))

    tmep.fit(train_x, train_y)

    #preds = rf.predict(valid_x)
    #pred_labels = np.rint(preds)
    accuracy = tmep.score(valid_x, valid_y)
    return accuracy

def drop_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   not_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   outliers_dropped = not_outliers.dropna().reset_index()

   return outliers_dropped

def cap_outliers_IQR(df,column):
    upper_limit = df[column].mean() + 3*df[column].std()
    lower_limit = df[column].mean() - 3*df[column].std()
    df[column] = np.where(df[column] > upper_limit,upper_limit,df[column])
    df[column] = np.where(df[column] < lower_limit,lower_limit,df[column])
    return df[column]

X = pd.read_csv('train1.csv',sep=';')
X_findAns = pd.read_csv('test1.csv',sep=';')
Id = X_findAns["id"]
tmp = X.columns
print(X.shape)
X=X.drop_duplicates(subset=tmp)
print(X.shape)
correlation = X.corr()
print(correlation['quality'].sort_values(ascending = False),'\n')

new = X['alcohol']
new = new*new - 5*X['density'] - 4*X['volatile acidity'] - 3*X['chlorides'] + 2*X['type']
new= new.rename('newFeature')
newf = X['density']
newf = -newf*newf -2*X['volatile acidity']*X['volatile acidity']
newf= newf.rename('newFeature2')
X=pd.concat([X,new],axis=1)
#X=pd.concat([X,newf],axis=1)
correlation = X.corr()
print(correlation['quality'].sort_values(ascending = False),'\n')

new1 = X_findAns['alcohol']
new1 = new1*new1 - 5*X_findAns['density'] - 4*X_findAns['volatile acidity'] - 3*X_findAns['chlorides'] + 2*X_findAns['type']
new1= new1.rename('newFeature')
newf1 = X_findAns['density']
newf1 = -newf1*newf1 -2*X_findAns['volatile acidity']*X_findAns['volatile acidity']
newf1= newf1.rename('newFeature2')
X_findAns=pd.concat([X_findAns,new1],axis=1)
#X_findAns=pd.concat([X_findAns,newf1],axis=1)

#X = X.drop(['density'],axis=1)
#X_findAns = X_findAns.drop(['density'],axis=1)

'''
for i in range(0,len(X.columns)):
    b=drop_outliers_IQR(X.iloc[:,i])
    b=b.drop(['index'],axis=1)
    if i==0:
        a=b
    else:
        a = pd.concat([a, b], axis=1)
print(len(X))
X=a.dropna().reset_index()
X=X.drop(['index'],axis=1)
print(len(X))
'''


for i in range(0,len(X.columns)):
    b=cap_outliers_IQR(X,X.columns[i])
    if i==0:
        a=b
    else:
        a = pd.concat([a, b], axis=1)
X=a.dropna().reset_index()
X=X.drop(['index'],axis=1)



#features, labels=  oversample.fit_resample(X.drop(["quality"],axis=1),X["quality"])

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

#{'eps': 0.003, 'n_alphas': 1000, 'max_iterLS': 3000, 'epsilon': 0.01, 'max_iterSVR': 3000, 'n_estimatorsRFT': 597, 'criterionRFT': 'squared_error', 'max_depthRFT': 7, 'min_samples_splitRFT': 6, 'min_samples_leafRFT': 5,
# 'n_estimatorsGBR': 121, 'learning_rateGBR': 0.1, 'max_depthGBR': 17, 'min_samples_splitGBR': 6, 'min_samples_leafGBR': 9, 'learning_rateXGB': 0.03, 'max_depthXGB': 15, 'n_estimatorsXGB': 716, 'subsample': 0.6}. Best is trial 20 with value: 0.45763137590974423.



estimators = [('ls', LassoCV(eps=0.001,n_alphas=200,max_iter=3000)),
              ('svr', LinearSVR(epsilon=0.01,max_iter=3000)),
              ('rfr',RandomForestRegressor(n_estimators=169,criterion='squared_error',max_depth=17,min_samples_split=10,min_samples_leaf=3)),
              ('GBR',GradientBoostingRegressor(n_estimators=825,learning_rate=0.1,max_depth=16,min_samples_split=3,min_samples_leaf=10)),
              ('Rid',RidgeCV())
              ]
'''
results, names = list(), list()
for name, model in estimators:
	scores = evaluate_model(model, scaled_df, labels)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
'''

reg = StackingRegressor(estimators=estimators,
                        final_estimator=xgb.XGBRegressor(learning_rate=0.03, max_depth=2,n_estimators=389,subsample=0.6))
'''
xxx = evaluate_model(reg, scaled_df, labels)
print('>%s %.3f (%.3f)' % ('stacking', np.mean(xxx), np.std(xxx)))
'''

X_train, X_test, y_train, y_test=train_test_split(scaled_df,labels,test_size=0.15,random_state=42)
'''
model = RandomForestRegressor(n_estimators=684,criterion='squared_error',max_depth=28,min_samples_split=2,min_samples_leaf=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.score(X_test,y_test))
'''

print(reg.fit(X_train,y_train).score(X_test,y_test))

'''
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
'''
ans = reg.predict(scaled_df1)
print(ans)


fname3 = 'submitFinal2.csv'
f=open(fname3,'w')
for i in range (-1,len(ans)):
    if (i==-1):
        f.write('id,quality')
    else:
        f.write( str(round(Id.iloc[i])) + ',' + str(ans[i]) )
    f.write('\n')
f.close()
