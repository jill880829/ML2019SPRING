import csv 
import numpy as np
import sys

w = np.load('model.npy')

fl2 = np.genfromtxt(sys.argv[1],dtype=str,delimiter=',',encoding='big5')
fl2 = np.delete(fl2,1,1)
fl2 = np.delete(fl2,0,1)
fl2_shape = fl2.shape
dat2 = np.zeros(fl2_shape)

for x in range(fl2_shape[0]):
	for y in range(fl2_shape[1]):
		if fl2[x,y] == 'NR':
			dat2[x,y] = float(0)
		else:
			dat2[x,y] = float(fl2[x,y])
			if dat2[x,y]<0:
				dat2[x,y] = dat2[x,y-1]

train_x2 = []
for i in range(240):
	tem_dat2 = dat2[i*18:(i+1)*18]
	new_x2 = np.append(tem_dat2.transpose().flatten(),[1])
	hypo_y = np.dot(new_x2,w)
	train_x2.append(hypo_y)

idnum = []
for i in range(240):
	idnum.append("id_"+str(i))
ans = []
ans.append(idnum)
ans.append(train_x2)
ttle = np.asarray([["id","value"]])
ans = np.asarray(ans)
ans = np.concatenate((ttle,ans.transpose()), axis = 0)
np.savetxt(sys.argv[2],ans,delimiter=',',fmt="%s")