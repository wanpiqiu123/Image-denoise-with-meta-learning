from config import *
from utils import *
from PIL import Image
from keras.optimizers import Adam
from keras.models import *
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

def test_img():
    # model_path = './unet_clean50-0.00.hdf5'
    # model_path = './unet2_clean50-0.00.hdf5'
    model_path = './unet2_clean50-0.01.hdf5'
    # model_path = './unet2_clean28-0.00.hdf5'
    # model_path = './unet_clean38-0.00.hdf5'

    test_model = load_model(model_path,{'m_psnr':m_psnr})
    # test_model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = [m_psnr])

    # test_model.summary()
    img = np.array(Image.open('./Kodak24_n/kodim01_1.png'))
    img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))

    # img = add_guassian(img,25)
    img = img.astype('float32') / 255.
    # print(img[0])

    result_img = test_model.predict(img)
    compared = np.concatenate([img[0,:,:,:],result_img[0,:,:,:]],1)
    compared = compared.astype('float32')*255.
    # compared = np.concatenate([img[0,:,:,:],g_img[0,:,:,:]],1)
    Image.fromarray(np.uint8(compared)).show()

    print("PSNR: "+str(psnr(img,result_img)))

def test_denoise():
    # model_path = './unet1_sigma25.hdf5'
    # model_path = './reptile_clean_sigma25_0.05.hdf5'
    model_path = './DnCNN_sigma15_6000.hdf5'
    test_model = load_model(model_path,{'m_psnr':m_psnr})
    img = np.array(Image.open('./Kodak24_256/img08_1.png'))
    img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
    g_img = add_guassian(img,25)
    g_img = g_img.astype('float32') / 255.
    img = img.astype('float32') / 255.
    result_img = test_model.predict(g_img)
    compared = np.concatenate([img[0,:,:,:],g_img[0,:,:,:],result_img[0,:,:,:]],1)
    compared = compared.astype('float32')*255.
    Image.fromarray(np.uint8(compared)).show()
    print("PSNR: "+str(psnr(img,result_img)))

def test_real_denoise():
    # model_path = './unet1_sigma25.hdf5'
    # model_path = './reptile_clean_sigma25_0.05.hdf5'
    # model_path = './guassian/DnCNN_sigma15_6000.hdf5'
    # model_path = './poisson/reptile_DnCNN_poisson_6000.hdf5'
    model_path = './guassian/unet0_sigma25_6000.hdf5'
    test_model = load_model(model_path,{'m_psnr':m_psnr})
    img = np.array(Image.open('./mean_256/img_mean15.png'))
    g_img = np.array(Image.open('./real_256/img_real15.png'))
    g_img = np.reshape(g_img, (1, IMG_H, IMG_W, NUM_C))
    g_img = g_img.astype('float32') / 255.
    # img = img.astype('float32') / 255.
    result_img = test_model.predict(g_img)
    result_img = np.clip(result_img,0,255)
    # compared = np.concatenate([img[0,:,:,:],g_img[0,:,:,:],result_img[0,:,:,:]],1)
    result_img = np.reshape(result_img,(IMG_H, IMG_W, NUM_C))
    result_img = result_img.astype('float32')*255.
    Image.fromarray(np.uint8(result_img)).show()
    print("PSNR: "+str(psnr(img,result_img)))

def test_n2c_record():
    img_dir = './BSD500_256/'
    # img_dir = './BSD500_256/'
    # csvname = "Unet0_25_6000_"+img_dir.split('/')[-2]+'.csv'
    # csvname = "reptile_DnCNN_25_"+img_dir.split('/')[-2]+'.csv'
    # csvname = "reptile_DnCNN_sigma50"+img_dir.split('/')[-2]+'.csv'
    csvname = "unet_sigma15_6000"+img_dir.split('/')[-2]+'.csv'
    # csvname = "reptile_0.1_sigma25_"+img_dir.split('/')[-2]+'.csv'
    csvfile = open(csvname, 'w', newline="")
    writer = csv.writer(csvfile,dialect='excel')
    writer.writerow(["NUM","c_PSNR","n_PSNR","c_SSIM","n_SSIM"])
    img_list = glob.glob(img_dir+'*.png')
    num_pic = len(img_list)
    i=0
    # model_path = './reptile_clean_sigma50_0.05.hdf5'
    # model_path = './reptile_DnCNN_sigma50.hdf5'
    # model_path = './poisson/reptile_unet_poisson_6000.hdf5'
    model_path = './unet_sigma15_6000.hdf5'
    # model_path = './reptile_DnCNN_sigma25.hdf5'
    test_model = load_model(model_path,{'m_psnr':m_psnr})
    for img_name in img_list:
        img = np.array(Image.open(img_name))
        img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
        # g_img = add_guassian(img,50)
        # g_img = g_img.astype('float32') / 255.
        img = img.astype('float32') / 255.
        g_img = add_poisson(img)
        c_img = test_model.predict(g_img)
        c_psnr = psnr(img,c_img)
        n_psnr = psnr(img,g_img)

        img = np.reshape(img,(IMG_H, IMG_W, NUM_C))
        c_img = np.reshape(c_img,(IMG_H, IMG_W, NUM_C))
        g_img = np.reshape(g_img,(IMG_H, IMG_W, NUM_C))
        c_ssim = ssim(img,c_img,multichannel=True)
        n_ssim = ssim(img,g_img,multichannel=True)
        writer.writerow([i,c_psnr,n_psnr,c_ssim,n_ssim])
        # compared = np.concatenate([img[0,:,:,:],g_img[0,:,:,:],c_img[0,:,:,:]],1)
        # compared = compared.astype('float32')*255.
        print(i)
        # Image.fromarray(np.uint8(compared)).show()
        i+=1

def compare_model():
    img_dir = './Kodak24_256/'
    csvname = "unet1_compare"+img_dir.split('/')[-2]+'.csv'
    csvfile = open(csvname, 'w', newline="")
    writer = csv.writer(csvfile,dialect='excel')
    writer.writerow(["NUM","n_PSNR","c_PSNR1","c_PSNR2"])
    img_list = glob.glob(img_dir+'*.png')
    num_pic = len(img_list)
    i=0
    model1_path = './unet0_sigma25.hdf5'
    model2_path = './unet0_sigma25_2.hdf5'
    test_model1 = load_model(model1_path,{'m_psnr':m_psnr})
    test_model2 = load_model(model2_path,{'m_psnr':m_psnr})
    for img_name in img_list:
        img = np.array(Image.open(img_name))
        img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
        g_img = add_guassian(img)
        g_img = g_img.astype('float32') / 255.
        img = img.astype('float32') / 255.
        c_img1 = test_model1.predict(g_img)
        c_img2 = test_model2.predict(g_img)
        c_psnr1 = psnr(img,c_img1)
        c_psnr2 = psnr(img,c_img2)
        n_psnr = psnr(img,g_img)
        writer.writerow([i,n_psnr,c_psnr1,c_psnr2])
        # compared = np.concatenate([img[0,:,:,:],g_img[0,:,:,:],c_img1[0,:,:,:],c_img2[0,:,:,:]],1)
        # compared = compared.astype('float32')*255.
        print(i)
        # Image.fromarray(np.uint8(compared)).show()
        i+=1

def record_test_result():
    img_dir = './Kodak24_n/'
    csvname = img_dir.split('/')[-2]+'.csv'
    csvfile = open(csvname, 'w', newline="")
    writer = csv.writer(csvfile,dialect='excel')
    writer.writerow(["NUM","c_PSNR","n_PSNR"])
    img_list = glob.glob(img_dir+'*.png')
    num_pic = len(img_list)
    # print(img_list)
    i=0

    c_model_path = './unet2_clean50-0.00.hdf5'
    n_model_path = './unet2_noise50-0.00.hdf5'
    c_model = load_model(c_model_path,{'m_psnr':m_psnr})
    n_model = load_model(n_model_path,{'m_psnr':m_psnr})
    for img_name in img_list:
        img = np.array(Image.open(img_name))
        img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
        img = add_guassian(img)
        img = img.astype('float32') / 255.
        c_img = c_model.predict(img)
        n_img = n_model.predict(img)
        c_psnr = psnr(img,c_img)
        n_psnr = psnr(img,n_img)
        writer.writerow([i,c_psnr,n_psnr])
        # compared = np.concatenate([img[0,:,:,:],c_img[0,:,:,:],n_img[0,:,:,:]],1)
        # compared = compared.astype('float32')*255.
        print(i)
        # Image.fromarray(np.uint8(compared)).show()
        i+=1

def all_model_n2c_poisson():
    file_list = glob.glob("./poisson/*.hdf5")
    img_dir = './BSD500_256/'
    img_list = glob.glob(img_dir+'*.png')
    num_pic = len(img_list)
    
    for file in file_list:
        useful_name = file.split("/")[-1].split(".")[0]
        training_num = useful_name.split("_")[-1]
        is_reptile = useful_name.split("_")[0]
        if is_reptile=="reptile":
            network_name = useful_name.split("_")[1]
            print(is_reptile+"_"+network_name+" starts!")
        else:
            network_name = is_reptile
            print(network_name+" starts!")
        csvname = useful_name+'.csv'
        # print(useful_name)
        # print(network_name)
        csvfile = open(csvname, 'w', newline="")
        writer = csv.writer(csvfile,dialect='excel')
        writer.writerow(["NUM","c_PSNR","n_PSNR","c_SSIM","n_SSIM"])
        model_path = file
        test_model = load_model(model_path,{'m_psnr':m_psnr})
        i=0
        all_c_psnr=all_c_ssim=all_n_psnr=all_n_ssim=0
        for img_name in img_list:
            img = np.array(Image.open(img_name))
            img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
            # g_img = add_guassian(img,50)
            # g_img = g_img.astype('float32') / 255.
            img = img.astype('float32') / 255.
            g_img = add_poisson(img)
            c_img = test_model.predict(g_img)
            c_psnr = psnr(img,c_img)
            n_psnr = psnr(img,g_img)
            all_c_psnr+=c_psnr
            all_n_psnr+=n_psnr

            img = np.reshape(img,(IMG_H, IMG_W, NUM_C))
            c_img = np.reshape(c_img,(IMG_H, IMG_W, NUM_C))
            g_img = np.reshape(g_img,(IMG_H, IMG_W, NUM_C))
            c_ssim = ssim(img,c_img,multichannel=True)
            n_ssim = ssim(img,g_img,multichannel=True)
            all_c_ssim+=c_ssim
            all_n_ssim+=n_ssim
            writer.writerow([i,c_psnr,n_psnr,c_ssim,n_ssim])
            i+=1
            if i%50==0:
                print(i)
        writer.writerow(["average",all_c_psnr/num_pic,all_n_psnr/num_pic,all_c_ssim/num_pic,all_n_ssim/num_pic])

def all_model_n2c_guassian():
    file_list = glob.glob("./guassian/*.hdf5")
    img_dir = './BSD500_256/'
    img_list = glob.glob(img_dir+'*.png')
    num_pic = len(img_list)
    # print(len(file_list))
    
    for file in file_list:
        useful_name = file.split("/")[-1].split(".")[0]
        is_reptile = useful_name.split("_")[0]
        
        if is_reptile=="reptile":
            network_name = useful_name.split("_")[1]
            sigma_name = useful_name.split("_")[2]
            # print(is_reptile+"_"+network_name+"_"+sigma_name+" starts!")
        else:
            network_name = is_reptile
            sigma_name = useful_name.split("_")[1]
            # print(network_name+"_"+sigma_name+" starts!")
        sigma_num = eval(sigma_name[-2:])
        # print(sigma_num)
        
        csvname = useful_name+"_"+img_dir.split("/")[-2]+'.csv'
        # print(csvname)
        print(useful_name)
        print(network_name)
        csvfile = open(csvname, 'w', newline="")
        writer = csv.writer(csvfile,dialect='excel')
        writer.writerow(["NUM","c_PSNR","n_PSNR","c_SSIM","n_SSIM"])
        model_path = file
        test_model = load_model(model_path,{'m_psnr':m_psnr})
        i=0
        all_c_psnr=all_c_ssim=all_n_psnr=all_n_ssim=0
        for img_name in img_list:
            img = np.array(Image.open(img_name))
            img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
            g_img = add_guassian(img,sigma_num)
            g_img = g_img.astype('float32') / 255.
            img = img.astype('float32') / 255.
            # g_img = add_poisson(img)
            c_img = test_model.predict(g_img)
            c_psnr = psnr(img,c_img)
            n_psnr = psnr(img,g_img)
            all_c_psnr+=c_psnr
            all_n_psnr+=n_psnr

            img = np.reshape(img,(IMG_H, IMG_W, NUM_C))
            c_img = np.reshape(c_img,(IMG_H, IMG_W, NUM_C))
            g_img = np.reshape(g_img,(IMG_H, IMG_W, NUM_C))
            c_ssim = ssim(img,c_img,multichannel=True)
            n_ssim = ssim(img,g_img,multichannel=True)
            all_c_ssim+=c_ssim
            all_n_ssim+=n_ssim
            writer.writerow([i,c_psnr,n_psnr,c_ssim,n_ssim])
            i+=1
            if i%50==0:
                print(i)
        writer.writerow(["average",all_c_psnr/num_pic,all_n_psnr/num_pic,all_c_ssim/num_pic,all_n_ssim/num_pic])
        
def all_model_n2c_real():
    file_list = glob.glob("./guassian/*.hdf5")
    img_dir = './mean_256/'
    n_dir = "./real_256/"
    img_list = glob.glob(img_dir+'*.png')
    num_pic = len(img_list)
    # print(len(file_list))

    # for img in img_list:
    #     print(n_dir+img.split("/")[-1])
    
    for file in file_list:
        useful_name = file.split("/")[-1].split(".")[0]
        csvname = useful_name+"_real"+'.csv'
        # print(csvname)
        print(useful_name)
        csvfile = open(csvname, 'w', newline="")
        writer = csv.writer(csvfile,dialect='excel')
        writer.writerow(["NUM","c_PSNR","n_PSNR","c_SSIM","n_SSIM"])
        model_path = file
        test_model = load_model(model_path,{'m_psnr':m_psnr})
        i=0
        all_c_psnr=all_c_ssim=all_n_psnr=all_n_ssim=0
        for img_name in img_list:
            img = np.array(Image.open(img_name))
            img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
            g_img_name=img_name.replace("mean","real")
            g_img = np.array(Image.open(g_img_name))
            g_img = np.reshape(g_img, (1, IMG_H, IMG_W, NUM_C))
            g_img = g_img.astype('float32') / 255.
            img = img.astype('float32') / 255.
            c_img = test_model.predict(g_img)
            c_psnr = psnr(img,c_img)
            n_psnr = psnr(img,g_img)
            all_c_psnr+=c_psnr
            all_n_psnr+=n_psnr

            img = np.reshape(img,(IMG_H, IMG_W, NUM_C))
            c_img = np.reshape(c_img,(IMG_H, IMG_W, NUM_C))
            g_img = np.reshape(g_img,(IMG_H, IMG_W, NUM_C))
            c_ssim = ssim(img,c_img,multichannel=True)
            n_ssim = ssim(img,g_img,multichannel=True)
            all_c_ssim+=c_ssim
            all_n_ssim+=n_ssim
            writer.writerow([i,c_psnr,n_psnr,c_ssim,n_ssim])
            i+=1
            if i%10==0:
                print(i)
        writer.writerow(["average",all_c_psnr/num_pic,all_n_psnr/num_pic,all_c_ssim/num_pic,all_n_ssim/num_pic])        

if __name__ == "__main__":
    # record_test_result()
    # test_img()
    # test_denoise()
    test_real_denoise()
    # test_n2c_record()
    # compare_model()
    # all_model_n2c_poisson()
    # all_model_n2c_guassian()
    # all_model_n2c_real()
