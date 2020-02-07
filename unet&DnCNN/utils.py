from config import *
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import glob
from PIL import Image
import matplotlib.pyplot as plt
import keras.backend as K
from skimage import util
import csv

# def m_psnr(y_true, y_predict):
#     # print(type(y_predict.eval()))
#     PSNR=SSIM=0
#     for j in range(NUM_C):
#         PSNR += psnr(np.uint8(y_true[:, :, :, j]), np.uint8(y_predict[:, :, :, j]))
#     PSNR/=NUM_C
#     return PSNR
def m_psnr(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


def load_data(dir_name,flag):
    file_list = glob.glob(dir_name+'*.png')
    num_pic = len(file_list)
    # print(num_pic)
    set = np.zeros((num_pic,IMG_H,IMG_W,NUM_C),dtype=np.float32)
    i=0
    if flag==True:
        file_name = "guassian"+str(num_pic)
    else:
        file_name = "clean"+str(num_pic)
    for file in file_list:
        img = np.array(Image.open(file))
        if img.shape != (IMG_H, IMG_W, NUM_C):
            print("error in "+file)
        else:
            img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
            if flag==True:
                img = img = add_guassian(img)
            img = img.astype('float32') / 255.
            set[i] = img
            i += 1
            if i % 1000 ==0:
                print("pic_num: "+str(i))

    print("save file to "+file_name+".npy")
    np.save(file_name,set) # The file is a 4D array and is normalized
    """
    #Show the image
    np.random.shuffle(set)
    plt.figure()
    plt.imshow(set[0].reshape(256,256,3))
    plt.show()
    """
    return set

def load_data_and_save(dir_name):
    file_list = glob.glob(dir_name+'*.png')
    num_pic = len(file_list)
    # print(num_pic)
    c_set = np.zeros((num_pic,IMG_H,IMG_W,NUM_C),dtype=np.float32)
    g_set = np.zeros((num_pic,IMG_H,IMG_W,NUM_C),dtype=np.float32)
    i=0
    for file in file_list:
        img = np.array(Image.open(file))
        if img.shape != (IMG_H, IMG_W, NUM_C):
            print("error in "+file)
        else:
            img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
            g_img = add_guassian(img,sigma=25)
            img = img.astype('float32') / 255.
            g_img = g_img.astype('float32') /255.
            c_set[i] = img
            g_set[i] = g_img
            i += 1
            if i % 1000 ==0:
                print("pic_num: "+str(i))
    g_file_name = "guassian_sigma25_"+str(num_pic)
    c_file_name = "clean_sigma25_"+str(num_pic)
    print("save file to "+c_file_name+".npy")
    np.save(c_file_name,c_set)
    print("save file to "+g_file_name+".npy")
    np.save(g_file_name,g_set)

def add_guassian(clean_set,sigma=SIGMA,normalized=False):
    if normalized==True:
        clean_set = clean_set.astype('float32')*255.
    noised_set = clean_set + np.random.normal(0, sigma, clean_set.shape)
    noised_set = np.clip(noised_set,0,255)
    if normalized==True:
        noised_set = noised_set.astype('float32')/255.
    return noised_set

def load_file(file_name):
    set = np.load(file_name)
    return set

def load_img(file_name,ADD_GUASSIAN):
    img = np.array(Image.open(file_name))
    img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
    if ADD_GUASSIAN==True:
        img = add_guassian(img)
    img = img.astype('float32') / 255.
    return img

def add_poisson(clean_set):
    noise_set=util.random_noise(clean_set,mode='poisson')
    return noise_set

def make_poisson(file_name):
    print("Read Clean File")
    clean_set = np.load(file_name)
    noise_set = np.zeros_like(clean_set)
    for i in range(len(clean_set)):
        noise_set[i]=util.random_noise(clean_set[i],mode='poisson')
        if i % 1000 ==0:
            print(i)
    p_file_name = "poisson_"+str(len(clean_set))
    print("Save Poisson File to "+p_file_name+".npy")
    np.save(p_file_name,noise_set)

def handle_csv():
    file_list = glob.glob("./useful_data/poisson/on_bsd/*.csv")
    csvname = "poisson_bsd.csv"
    csvfile = open(csvname, 'w', newline="")
    writer = csv.writer(csvfile,dialect='excel')
    writer.writerow(["model","c_PSNR","n_PSNR","c_SSIM","n_SSIM"])
    for file in file_list:
        head_name = "_".join(".".join(file.split("/")[-1].split(".")[:-1]).split("_")[:-1])
        with open(file) as csvfile:
            mLines = csvfile.readlines()
        targetLine = mLines[-1]
        line = targetLine.split(",")[1:]
        line.insert(0,head_name)
        writer.writerow(line)
        # print(head_name)
        # a=targetLine.split(',')[0]
        # print(targetLine)

if __name__=='__main__':
    # load_data_and_save('./color_30000/')
    # make_poisson("./clean24.npy")
    # load_data('./Kodak24_n/',NOISE_FLAG)
    # n_img = load_img("./Kodak24_256/img05_1.png",True)
    # n_img = np.reshape(n_img,(256,256,3))
    # n_img = n_img.astype('float32') * 255.
    # Image.fromarray(np.uint8(n_img)).show()
    handle_csv()