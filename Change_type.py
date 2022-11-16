fname = 'train.csv'
fh = open(fname)
b=list()
c=list()
for line in fh:
        a = line.strip().split(';')
        b.append(a)

for i in range(len(b)):
    for j in range(len(b[i])):
        if (b[i][j] == 'white'):
            b[i][j] = 1
        if (b[i][j] == 'red'):
            b[i][j] = 0

f = open("train1.csv", "w")
for i in range(len(b)):
    for j in range(len(b[i])):
        if (j==len(b[i])-1):
            if(i==0):
                f.write(str(b[i][j])+'\n')
            else:
                f.write(str(b[i][j])+'\n')
        else:
            f.write(str(b[i][j])+';')

f.close()