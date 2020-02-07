import tensorflow as tf
from config import *
from networks import vgg16, FormResNet
from ops import sobel
import os,csv
from PIL import Image
import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage import util


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
        saver = tf.train.Saver()
        # saver.restore(self.sess, "./save_para_3_sigma25_2/FormResNet25.ckpt")
        saver.restore(self.sess, "./sigma50_6000/FormResNet50.ckpt")
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())

    def train(self):
        filepath = "./TrainingSet//"
        filenames = os.listdir(filepath)
        saver = tf.train.Saver()
        for epoch in range(EPOCHS):
            for i in range(filenames.__len__()//BATCH_SIZE):
                cleaned_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C], dtype=np.float32)
                for idx, filename in enumerate(filenames[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]):
                    cleaned_batch[idx, :, :, 0] = np.array(Image.open(filepath+filename))
                noised_batch = cleaned_batch + np.random.normal(0, SIGMA, cleaned_batch.shape)
                noised_batch = np.clip(noised_batch,0,255)
                self.sess.run(self.Opt, feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: True})
                if i % 10 == 0:
                    [loss, denoised_img] = self.sess.run([self.L_cross, self.denoised_img], feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: False})
                    PSNR = psnr(np.uint8(cleaned_batch[0, :, :, 0]), np.uint8(denoised_img[0, :, :, 0]))
                    SSIM = ssim(np.uint8(cleaned_batch[0, :, :, 0]), np.uint8(denoised_img[0, :, :, 0]))
                    print("Epoch: %d, Step: %d, Loss: %g, psnr: %g, ssim: %g"%(epoch, i, loss, PSNR, SSIM))
                    compared = np.concatenate((cleaned_batch[0, :, :, 0], noised_batch[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
                    Image.fromarray(np.uint8(compared)).save("./TrainingResults//"+str(epoch)+"_"+str(i)+".jpg")
                if i % 500 == 0:
                    saver.save(self.sess, "./save_para_3_sigma25//FormResNet.ckpt")
            np.random.shuffle(filenames)

    def test(self, cleaned_path="./Kodak24_256//img02_1.png"):
        cleaned_img = np.reshape(np.array(Image.open(cleaned_path), dtype=np.float32), [1, 256, 256, 3])
        noised_img = cleaned_img + np.random.normal(0, SIGMA, cleaned_img.shape)
        n_img = cleaned_img + np.random.normal(0, 15, cleaned_img.shape)
        compared=np.concatenate((n_img[0, :, :, :], noised_img[0, :, :, :]), 1)
        Image.fromarray(np.uint8(compared)).show()
        # n_PSNR=n_SSIM=0
        # for i in range(3):
        #     n_PSNR += psnr(np.uint8(cleaned_img[0, :, :, i]), np.uint8(noised_img[0, :, :, i]))
        #     n_SSIM += ssim(np.uint8(cleaned_img[0, :, :, i]), np.uint8(noised_img[0, :, :, i]))
        # n_PSNR/=3.0
        # n_SSIM/=3.0
        # print("n_psnr: %g, n_ssim: %g" % (n_PSNR, n_SSIM))
        # [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        # PSNR=SSIM=0
        # for i in range(3):
        #     PSNR += psnr(np.uint8(cleaned_img[0, :, :, i]), np.uint8(denoised_img[0, :, :, i]))
        #     SSIM += ssim(np.uint8(cleaned_img[0, :, :, i]), np.uint8(denoised_img[0, :, :, i]))
        # PSNR/=3.0
        # SSIM/=3.0
        # print("psnr: %g, ssim: %g" % (PSNR, SSIM))
        # compared = np.concatenate((cleaned_img[0, :, :, :], noised_img[0, :, :, :], denoised_img[0, :, :, :]), 1)
        # Image.fromarray(np.uint8(compared)).show()

    def test_all(self, cleaned_path="./BSD500_256//"):
        filenames = os.listdir(cleaned_path)
        csvname = 'sigma50_bsd_6000.csv'
        csvfile = open(csvname, 'w', newline="")
        writer = csv.writer(csvfile,dialect='excel')
        writer.writerow(["num","PSNR","n_PSNR","c_SSIM","n_SSIM"])
        idx = 0
        tn_PSNR=tn_SSIM=tc_PSNR=tc_SSIM=0
        for f in filenames:
            idx+=1
            path = cleaned_path+f
            cleaned_img = np.reshape(np.array(Image.open(path), dtype=np.float32), [1, 256, 256, 3])
            noised_img = cleaned_img + np.random.normal(0, SIGMA, cleaned_img.shape)
            noised_img = np.clip(noised_img,0,255)
            # c_batch = cleaned_img.astype('float32')/255.
            # noised_img=util.random_noise(c_batch,mode='poisson')
            # noised_img = noised_img.astype('float32')*255.
            n_PSNR=n_SSIM=0
            for i in range(3):
                n_PSNR += psnr(np.uint8(cleaned_img[0, :, :, i]), np.uint8(noised_img[0, :, :, i]))
                n_SSIM += ssim(np.uint8(cleaned_img[0, :, :, i]), np.uint8(noised_img[0, :, :, i]))
            n_PSNR/=3.0
            n_SSIM/=3.0
            tn_PSNR+=n_PSNR
            tn_SSIM+=n_SSIM
            # print("n_psnr: %g, n_ssim: %g" % (n_PSNR, n_SSIM))
            [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
            PSNR=SSIM=0
            for i in range(3):
                PSNR += psnr(np.uint8(cleaned_img[0, :, :, i]), np.uint8(denoised_img[0, :, :, i]))
                SSIM += ssim(np.uint8(cleaned_img[0, :, :, i]), np.uint8(denoised_img[0, :, :, i]))
            PSNR/=3.0
            SSIM/=3.0
            tc_PSNR+=PSNR
            tc_SSIM+=SSIM
            # print("psnr: %g, ssim: %g" % (PSNR, SSIM))
            writer.writerow([idx,PSNR,n_PSNR,SSIM,n_SSIM])
        tn_PSNR/=idx
        tn_SSIM/=idx
        tc_PSNR/=idx
        tc_SSIM/=idx
        writer.writerow(["average",tc_PSNR,tn_PSNR,tc_SSIM,tn_SSIM])

if __name__ == "__main__":
    m = Main()
    # m.test()
    m.test_all()
