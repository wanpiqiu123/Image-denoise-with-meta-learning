from utils import *
from config import *
from network import *

data_c = load_file("./clean24.npy")
data_n = load_file("./guassian24.npy")
combine = np.concatenate((data_c[5, :, :, :], data_n[5, :, :, :]), 1)
combine = combine.astype('float32')*255.
img = Image.fromarray(np.uint8(combine)).save("combine.png")
