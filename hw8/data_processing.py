import numpy as np
from keras.models import load_model

model = load_model('best2.h5')
weight = model.get_weights()
save_weight = []
for x in weight:
	save_weight.append(x.ravel())

save_weight = np.concatenate(save_weight)
save_weight = save_weight.astype('float16')
np.savez_compressed('weight2.npz',save_weight)