from utils import load_file,add_poisson
from config import *
# from network import *
from skimage import util
import numpy as np
from PIL import Image

# data = load_file("./guassian24.npy")
# data = load_file("./em_clean24.npy")
data = load_file("./clean24.npy")
# data = data.astype('float32')*255
# print(data.shape)
n_data = add_poisson(data)
print(n_data)
# for i in range(len(data)):
#     img = data[i,:,:,:].reshape(IMG_H, IMG_W, NUM_C)
#     img = img.astype('float32')*255.
#     if i<=10:
#         Image.fromarray(np.uint8(img)).show()

# mlp = MLP()
# mlp.compile(optimizer = Adam(lr = 1e-4), loss = 'mse')
# mlp.summary()

# img = np.array(Image.open("./Kodak24_256/img02_1.png"))
# img = np.reshape(img,(1,256,256,3))
# # img = img.astype('float32') / 255
# n_img=util.random_noise(img,mode='poisson')
# # img = img.astype('float32') * 255
# n_img = n_img.astype('float32') * 255
# n_img = np.reshape(n_img,(256,256,3))
# Image.fromarray(np.uint8(n_img)).show()
# compared = np.concatenate([img,n_img],1)
# print(img)
# print(n_img)

# Image.fromarray(np.uint8(compared)).show()

# clean_set = np.load("./clean24.npy")
# noise_set = np.zeros_like(clean_set)
# for i in range(len(clean_set)):
#     noise_set[i]=util.random_noise(clean_set[i],mode='poisson')
#     img = np.reshape(noise_set[i],(256,256,3)).astype('float32') * 255
#     Image.fromarray(np.uint8(img)).show()
# print(len(clean_set))