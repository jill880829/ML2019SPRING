import os
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import  Adam
# from keras.callbacks import ModelCheckpoint,EarlyStopping
# from keras.models import load_model

x_train = np.genfromtxt(sys.argv[1],delimiter=',')
x_train = np.delete(x_train,0,0)
x_train = np.delete(x_train,np.s_[32:39],1)
p = np.random.permutation(x_train.shape[0])
x_train = x_train[p]
x_train_n = (x_train-x_train.min(axis=0)) / (x_train.max(axis=0)-x_train.min(axis=0))

y_train = np.genfromtxt(sys.argv[2],delimiter=',')
y_train = np.delete(y_train,0,0)
y_train = y_train[p]

x_test = np.genfromtxt(sys.argv[3],delimiter=',')
x_test = np.delete(x_test,0,0)
x_test = np.delete(x_test,np.s_[32:39],1)
x_test = (x_test-x_train.min(axis=0)) / (x_train.max(axis=0)-x_train.min(axis=0))

model = GradientBoostingClassifier(max_depth=6)
model.fit(x_train_n,y_train)
predict = model.predict(x_test)
predict = predict.tolist()
for i in range(len(predict)):
	predict[i] = int(predict[i])
# model = Sequential()
# model.add(Dense(units=2500,input_dim=x_train_n.shape[1],kernel_initializer='random_normal',activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(units=100,activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(units=1,activation='sigmoid'))
# model.summary()
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(x=x_train_n,y=y_train,epochs=70,batch_size=256,validation_split = 0.2,verbose=2)
# score = model.evaluate(x_train_n,y_train)
# print(score[1])

# predict = np.squeeze(model.predict(x_test))
# predict = predict.tolist()
# for i in range(len(predict)):
# 	if predict[i]>0.5:
# 		predict[i]=1
# 	else:
# 		predict[i]=0

idnum = []
for i in range(x_test.shape[0]):
	idnum.append(i+1)
ans = []
ans.append(idnum)
ans.append(predict)
ttle = np.asarray([["id","label"]])
ans = np.asarray(ans)
ans = np.concatenate((ttle,ans.transpose()), axis = 0)
np.savetxt(sys.argv[4],ans,delimiter=',',fmt="%s")
