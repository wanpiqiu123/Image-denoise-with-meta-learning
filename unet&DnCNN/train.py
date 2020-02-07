from network import *
from config import *
from utils import *
from keras.callbacks import ModelCheckpoint

# original_img_dir = './color_30000/'
original_img_dir = './color_6000_256/'
# original_img = load_data(original_img_dir,NOISE_FLAG)
# original_img = load_file("./clean24.npy")
original_img = load_file("./clean30486.npy")
#original_img = load_data(original_img_dir,NOISE_FLAG)
noise_img = load_file("./guassian30486.npy")

# if NOISE_FLAG==True:
#     original_img = add_guassian(original_img)
# print(original_img.shape)

# unet = Unet()
unet = Unet()
unet.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = [m_psnr])

if NOISE_FLAG==False:
    model_checkpoint = ModelCheckpoint('unet0_clean{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True,mode='min')
if NOISE_FLAG==True:
    model_checkpoint = ModelCheckpoint('unet0_noise{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True,mode='min')
unet.fit(original_img,original_img,BATCH_SIZE,EPOCHS,1,validation_split=0.3,callbacks=[model_checkpoint])

