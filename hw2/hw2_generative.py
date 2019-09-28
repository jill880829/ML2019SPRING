# import sys
# import csv
# import math
# import numpy as np
# from numpy.linalg import inv

# x_train = np.genfromtxt(sys.argv[1],delimiter=',')
# x_train = np.delete(x_train,0,0)
# x_train_n = x_train / x_train.max(axis=0)
# y_train = np.genfromtxt(sys.argv[2],delimiter=',')
# y_train = np.delete(y_train,0,0)
# x_test = np.genfromtxt(sys.argv[3],delimiter=',')
# x_test = np.delete(x_test,0,0)
# x_test = x_test / x_train.max(axis=0)

# dat1 = []
# dat2 = []
# for i in range(len(y_train)):
# 	if y_train[i] == 0:
# 		dat1.append(x_train_n[i])
# 	else:
# 		dat2.append(x_train_n[i])
# dat1 = np.asarray(dat1)
# dat2 = np.asarray(dat2)
# u1 = np.mean(dat1,axis = 0)
# u2 = np.mean(dat2,axis = 0)
# sig1 = np.cov(dat1.T)
# sig2 = np.cov(dat2.T)
# sz1 = dat1.shape[0]
# sz2 = dat2.shape[0]
# sig = sig1*(sz1/(sz1+sz2))+sig2*(sz2/(sz1+sz2))
# w = np.transpose(np.dot(np.transpose(u1-u2),inv(sig)))
# np.savetxt(sys.argv[5],inv(sig),delimiter=',',fmt="%s")
# b = -1/2*np.dot(np.transpose(u1),np.dot(inv(sig),u1))+1/2*np.dot(np.transpose(u2),np.dot(inv(sig),u2))+math.log(sz1/sz2)
# predict = []
# for i in range(x_test.shape[0]):
# 	res = np.dot(w,x_test[i])+b
# 	if res<=0:
# 		predict.append(0)
# 	else:
# 		predict.append(1)

# idnum = []
# for i in range(x_test.shape[0]):
# 	idnum.append(i+1)
# ans = []
# ans.append(idnum)
# ans.append(predict)
# ttle = np.asarray([["id","label"]])
# ans = np.asarray(ans)
# ans = np.concatenate((ttle,ans.transpose()), axis = 0)
# np.savetxt(sys.argv[4],ans,delimiter=',',fmt="%s")

import numpy as np
import sys
import csv  
from numpy.linalg import inv

class data_manager():
    def __init__(self):
        self.data = {}  
    
    def read(self,name,path):
        with open(path,newline = '') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:] ,dtype = float)  
            if name == 'X_train':
                self.mean = np.mean(rows,axis = 0).reshape(1,-1)
                self.std = np.std(rows,axis = 0).reshape(1,-1)
                self.theta = np.ones((rows.shape[1] + 1,1),dtype = float) 
                for i in range(rows.shape[0]):
                    rows[i,:] = (rows[i,:] - self.mean) / self.std  

            elif name == 'X_test': 
                for i in range(rows.shape[0]):
                    rows[i,:] = (rows[i,:] - self.mean) / self.std 

            self.data[name] = rows  

    def find_theta(self):
        class_0_id = []
        class_1_id = []
        for i in range(self.data['Y_train'].shape[0]):
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)

        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id] 

        mean_0 = np.mean(class_0,axis = 0)
        mean_1 = np.mean(class_1,axis = 0)  

        n = class_0.shape[1]
        cov_0 = np.zeros((n,n))
        cov_1 = np.zeros((n,n))
        
        for i in range(class_0.shape[0]):
            cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [(class_0[i] - mean_0)]) / class_0.shape[0]

        for i in range(class_1.shape[0]):
            cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [(class_1[i] - mean_1)]) / class_1.shape[0]

        cov = (cov_0*class_0.shape[0] + cov_1*class_1.shape[0]) / (class_0.shape[0] + class_1.shape[0])
 
        self.w = np.transpose(((mean_0 - mean_1)).dot(inv(cov)) )
        self.b =  (- 0.5)* (mean_0).dot(inv(cov)).dot(mean_0)\
            + 0.5 * (mean_1).dot(inv(cov)).dot(mean_1)\
            + np.log(float(class_0.shape[0]) / class_1.shape[0]) 

        result = self.func(self.data['X_train'])
        answer = self.predict(result)


    def func(self,x):
        arr = np.empty([x.shape[0],1],dtype=float)
        for i in range(x.shape[0]):
            z = x[i,:].dot(self.w) + self.b
            z *= (-1)
            arr[i][0] = 1 / (1 + np.exp(z))
        return np.clip(arr, 1e-8, 1-(1e-8))

    def predict(self,x):
        ans = np.ones([x.shape[0],1],dtype=int)
        for i in range(x.shape[0]):
            if x[i] > 0.5:
                ans[i] = 0; 
        return ans

    def write_file(self,path):
        result = self.func(self.data['X_test'])
        answer = self.predict(result)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow(['id','label']) 
            for i in range(answer.shape[0]):
                writer.writerow([i+1,answer[i][0]])

dm = data_manager()
dm.read('X_train',sys.argv[1])
dm.read('Y_train',sys.argv[2])
dm.read('X_test',sys.argv[3])
dm.find_theta()
dm.write_file(sys.argv[4])
