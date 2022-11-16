import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
import math

from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer


def meanCalc(matrix):
    sum = 0
    for i in matrix:
        sum = sum + i
    return sum/(len(matrix))

def sigmaCalc(matrix, mean):
    sumSquare = 0
    for i in matrix:
        sumSquare = sumSquare + i*i
    return math.sqrt(sumSquare/(len(matrix)) - mean*mean)

def Normalize(X):
    mu=list()
    sigma=list()
    X_norm=list()
    for i in range (0, np.shape(X)[1]):
        mu.append( meanCalc(X[:, i]) )
        sigma.append( sigmaCalc(X[:, i],mu[i]) )

    for i in range (0,np.shape(X)[1]):
        temp=list()
        for j in range (0,np.shape(X)[0]):
            temp.append( (X[j, i] - mu[i] ) / sigma[i] );
        X_norm.append(temp)
    return np.transpose(np.array(X_norm)).tolist(),mu,sigma


def change_type(b):
    for i in range(len(b)):
        for j in range(len(b[i])):
            if (b[i][j] == 'white'):
                b[i][j] = 1
            if (b[i][j] == 'red'):
                b[i][j] = 0

def check_dup(X,XTest):
    check = [0]*len(X)
    #check = np.zeros( (np.shape(X)[0],np.shape(XTest)[0]))
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(XTest)[0]):
            if (np.alltrue(X[i]==XTest[j])):
                check[i]=1
    return check

def check_different(X):
    check = [0]*len(X)
    count=0
    for i in range(0, len(X[0]) ):
        cal=list()
        cal.append(mu[i] + 3*sigma[i])
        cal.append(mu[i] - 3*sigma[i])
        for j in range(0, len(X) ):
            if(check[j]!=1):
                if(X[j][i]> cal[0] or X[j][i] < cal[1] ):
                    check[j]=1
                    count+=1
    return check


print('Loading train data ...\n');
fname = 'train.csv'
fh = open(fname)
b=list()
for line in fh:
        a = line.strip().split(';')
        b.append(a[:])

data = np.array(b[1:])
change_type(data)
data = data.astype('float64')

m = len(data);
print(m)
headline = b[:][0]
X_train = data[:int(0.8*m), [0,1,2,3,4,5,6,7,8,9,10]]
y_train = data[:int(0.8*m), 11]

X_valid = data[int(0.8*m):, [0,1,2,3,4,5,6,7,8,9,10]]
y_valid = data[int(0.8*m):, 11]
m = len(X_train)

print('Loading test data ...\n');
fname2 = 'test.csv'
fh = open(fname2)
c=list()
for line in fh:
        f = line.strip().split(';')
        c.append(f[:])

dataTest = np.array(c[1:])
change_type(dataTest)
dataTest = dataTest.astype('float64')

ID = dataTest[:, 0]
XTest = dataTest[:, [1,2,3,4,5,6,7,8,9,10,11]]

a,mu,sigma = Normalize(X_train);

check1 =check_dup(X_train,XTest)
check2 = check_different(X_train)

X=np.array(a)

'''
newCheck=list()
count=0
for i in range(0,m):
    if(check1[i]==1 or check2[i]==1):
        count+=1
        newCheck.append(i)
print(count)
X_train = np.delete(X_train,newCheck,0)
y_train = np.delete(y_train,newCheck,0)
m = len(y_train)
'''

# create regressor object
regressor = RandomForestRegressor(n_estimators=1000, random_state=42)



# fit the regressor with x and y data
regressor.fit(X_train, y_train)
ans = regressor.predict(XTest)
print(regressor.score(X_valid,y_valid))


fname3 = 'submitNewModel.csv'
f=open(fname3,'w')
for i in range (-1,len(ans)):
    if (i==-1):
        f.write('id,quality')
    else:
        f.write( str(round(ID[i])) + ',' + str(ans[i]) )
    f.write('\n')
f.close()
