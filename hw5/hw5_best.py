import sys
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, \
							   resnet101, densenet121, densenet169 

def read(idx):
	img_path = sys.argv[1]+'/%03d.png'%(idx)
	img = Image.open(img_path)
	return img

def save(idx,img):
	img_path = sys.argv[2]+'/%03d.png'%(idx)
	img.save(img_path)

# using pretrain proxy model, ex. VGG16, VGG19...
model = resnet50(pretrained=True)
# or load weights from .pt file
# model = torch.load_state_dict(...)
# use eval mode
model.eval()

# loss criterion
criterion = nn.CrossEntropyLoss()

for i in range(200):
	image = read(i)
	# you can do some transform to the image, ex. ToTensor()
	pxl_mean = np.asarray([0.485, 0.456, 0.406])
	pxl_std = np.asarray([0.229, 0.224, 0.225])
	trans = transform.Compose([transform.ToTensor(),transform.Normalize(mean=pxl_mean,
								 std=pxl_std)])
	image = trans(image)
	image = image.unsqueeze(0)
	epsilon = 0.4/255/0.23
	epoch = 5
	target_label = model(image)
	target_label = torch.tensor([torch.argmax(target_label)])
	for j in range(epoch):
		image.requires_grad = True
		zero_gradients(image)
		output = model(image)
		loss = criterion(output, target_label)
		loss.backward() 
		# add epsilon to image
		image = image + epsilon * image.grad.sign_()
		image = image.detach()

	# do inverse_transform if you did some transformation
	inv_trans = transform.Normalize(mean=-pxl_mean/pxl_std,std=1/pxl_std)(image[0])
	inv_trans = torch.clamp(inv_trans,0,1)
	inv_trans = transform.ToPILImage()(inv_trans)
	save(i,inv_trans)