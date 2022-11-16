import math

import numpy as np

#import statistics as sta,Eva Elfie

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
                check[i][j]=1
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

def computeCost(X, y, theta):
    J=0
    for i in range(0,m):
        err = np.dot(np.transpose(theta) , X[i]) - y[i]
        J += err*err;
    J /= (2 * m);
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history=list()
    for iter in range(0,num_iters):
        total = np.zeros(np.shape(X)[1])
        for i in range(0,m):
            err = np.dot(np.transpose(theta) , X[i]) - y[i]
            total += err * X[i];

        theta -= alpha * (1 / m) * total;

        J_history.append(computeCost(X, y, theta))
    return theta,J_history

def calTheta(X,y):
    tmp =  np.linalg.inv(np.matmul(np.transpose(X),X))
    tmp2 = np.matmul(tmp,np.transpose(X))
    tmp3 = np.matmul(tmp2,y)
    #theta_best_values = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return tmp3

def calPredict(XTest, theta):
    ans=list()
    for i in range(0,np.shape(XTest)[0]):
        err=0
        for j in range(0,np.shape(XTest)[1]):
            err+= XTest[i][j] * theta[j]
        ans.append(err)
    return ans


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

headline = b[:][0]
X = data[:, [0,1,2,3,4,5,6,7,8,9,10,12]]
y = data[:, 11]
m = len(y);

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
XTest = dataTest[:, [1,2,3,4,5,6,7,8,9,10,11,12]]

check1 =check_dup(X,XTest)

print('Normalizing Features ...\n');
a,mu,sigma = Normalize(X);
check2 = check_different(X)

X = np.array(a)

newCheck=list()
count=0
for i in range(0,m):
    if(check1[i]==1 or check2[i]==1):
        count+=1
        newCheck.append(i)
print(count)

X=np.delete(X,newCheck,0)
y = np.delete(y,newCheck,0)
m = len(y)
X=np.column_stack( (np.ones( np.shape(X)[0] ),X) )


print('Running gradient descent ...\n');

alpha = 0.3;
num_iters = 2000;
theta = np.zeros( np.shape(X)[1] );

theta, J_history = gradientDescent(X, y, theta, alpha, num_iters);
print(J_history)
realTheta = calTheta(X,y)



print(theta)
print(realTheta)

XTest,mu1,sigma1 = Normalize(XTest)

XTest = (np.column_stack((np.ones(np.shape(XTest)[0]),XTest)))

answer = calPredict(XTest,theta)


fname3 = 'submit.csv'
f=open(fname3,'w')
for i in range (-1,len(answer)):
    if (i==-1):
        f.write('id,quality')
    else:
        f.write( str(round(ID[i])) + ',' + str(answer[i]) )
    f.write('\n')
f.close()


