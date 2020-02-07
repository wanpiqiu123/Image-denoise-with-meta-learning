import tensorflow as tf
from config import *
from networks import vgg16, FormResNet
from ops import sobel
import os
import csv,time
from PIL import Image
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim


class Main:
    def __init__(self):
        self.clean_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.noised_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)
        form_resnet = FormResNet("FormResNet")
        self.denoised_img, self.res = form_resnet(self.noised_img, self.train_phase)
        self.L_pix = tf.reduce_mean(tf.reduce_sum(tf.square(self.denoised_img - self.clean_img), [1, 2, 3]))
        # self.Phi = vgg16(tf.concat([self.denoised_img, self.denoised_img, self.denoised_img], 3))
        # self.Phi_ = vgg16(tf.concat([self.clean_img, self.clean_img, self.clean_img], 3))
        self.Phi = vgg16(self.denoised_img)
        self.Phi_ = vgg16(self.clean_img)
        self.L_feat = tf.reduce_mean(tf.square(self.Phi - self.Phi_))
        self.L_grad = tf.reduce_mean(tf.reduce_sum(tf.abs(sobel(self.denoised_img)[0] - sobel(self.clean_img)[0]) +\
                      tf.abs(sobel(self.denoised_img)[1] - sobel(self.clean_img)[1]), [1, 2, 3]))
        self.L_cross = (1 - ALPHA - BETA) * self.L_pix + ALPHA * self.L_feat + BETA * self.L_grad
        self.Opt = tf.train.AdamOptimizer(1e-4).minimize(self.L_cross)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        filepath = "./color_pic//"
        filenames = os.listdir(filepath)
        saver = tf.train.Saver()
        csvname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.csv'
        csvfile = open(csvname, 'w', newline="")
        writer = csv.writer(csvfile,dialect='excel')
        writer.writerow(["Epoch","Step","Loss","PSNR","SSIM"])

        best_PSNR=0
        for epoch in range(EPOCHS):
            total_PSNR = 0
            num_iter = filenames.__len__()//BATCH_SIZE//100
            for i in range(filenames.__len__()//BATCH_SIZE):
                cleaned_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C], dtype=np.float32)
                for idx, filename in enumerate(filenames[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]):
                    cleaned_batch[idx, :, :, :] = np.array(Image.open(filepath+filename))
                noised_batch = cleaned_batch + np.random.normal(0, SIGMA, cleaned_batch.shape)
                self.sess.run(self.Opt, feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: True})
                if i % 100 == 0:
                    [loss, denoised_img] = self.sess.run([self.L_cross, self.denoised_img], feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: False})
                    PSNR=SSIM=0
                    for j in range(3):
                        PSNR += psnr(np.uint8(cleaned_batch[0, :, :, j]), np.uint8(denoised_img[0, :, :, j]))
                        SSIM += ssim(np.uint8(cleaned_batch[0, :, :, j]), np.uint8(denoised_img[0, :, :, j]))
                    PSNR/=3.0
                    total_PSNR+=PSNR
                    SSIM/=3.0
                    print("Epoch: %d, Step: %d, Loss: %g, psnr: %g, ssim: %g"%(epoch, i, loss, PSNR, SSIM))
                    data = [epoch, i, loss, PSNR, SSIM]
                    writer.writerow(data)
                    compared = np.concatenate((cleaned_batch[0, :, :, 0], noised_batch[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
                    Image.fromarray(np.uint8(compared)).save("./TrainingResults//"+str(epoch)+"_"+str(i)+".jpg")
            total_PSNR/=num_iter
            if total_PSNR>best_PSNR:
                print("save_para!")
                saver.save(self.sess, "./save_para//FormResNet"+str(SIGMA)+".ckpt")
                best_PSNR=total_PSNR
            np.random.shuffle(filenames)

    def test(self, cleaned_path="./crop_color_test//img24901_1.png"):
        cleaned_img = np.reshape(np.array(Image.open(cleaned_path), dtype=np.float32), [1, 256, 256, 3])
        noised_img = cleaned_img + np.random.normal(0, SIGMA, cleaned_img.shape)
        [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        # PSNR = psnr(np.uint8(cleaned_img[0, :, :, :]), np.uint8(denoised_img[0, :, :, :]))
        # SSIM = ssim(np.uint8(cleaned_img[0, :, :, :]), np.uint8(denoised_img[0, :, :, :]))
        PSNR=SSIM=0
        for i in range(3):
            PSNR += psnr(np.uint8(cleaned_img[0, :, :, i]), np.uint8(denoised_img[0, :, :, i]))
            SSIM += ssim(np.uint8(cleaned_img[0, :, :, i]), np.uint8(denoised_img[0, :, :, i]))
        PSNR/=3.0
        SSIM/=3.0
        print("psnr: %g, ssim: %g" % (PSNR, SSIM))
        compared = np.concatenate((cleaned_img[0, :, :, :], noised_img[0, :, :, :], denoised_img[0, :, :, :]), 1)
        Image.fromarray(np.uint8(compared)).show()


if __name__ == "__main__":
    m = Main()
    m.train()
