import numpy as np

def change_type(b):
    for i in range(len(b)):
        for j in range(len(b[i])):
            if (b[i][j] == 'white'):
                b[i][j] = 1
            if (b[i][j] == 'red'):
                b[i][j] = 0

print('Loading test data ...\n');
fname2 = 'test.csv'
fh = open(fname2)
c=list()
for line in fh:
        f = line.strip().split(';')
        c.append(f[:])
label = np.array(c[0])
dataTest = np.array(c[1:])
change_type(dataTest)
dataTest = dataTest.astype('float64')
print(label)
print(dataTest[0])


fname3 = 'test1.csv'
f=open(fname3,'w')
for i in range (-1,len(dataTest)):
    if (i==-1):
        for j in range(0,len(label)):
            if j==len(label)-1:
                f.write(label[j])
            else:
                f.write(label[j]+';')
    else:
        for j in range(0, len(dataTest[0])):
            if j == len(dataTest[0])-1:
                f.write(str(dataTest[i][j]))
            else:
                f.write(str(dataTest[i][j])+';')
    f.write('\n')
f.close()
