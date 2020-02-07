from utils import *
from network import Unet
from keras.models import *
from keras.optimizers import Adam

def embedding_layer(unet_model,img):
    unet_model.summary()
    embedding = Model(inputs=unet_model.input, outputs = unet_model.get_layer("conv2d_5").output)
    predict = embedding.predict(img)
    # pridict=np.reshape(predict,(16,16,4))
    # for i in range(4):
    # np.set_printoptions(threshold=np.inf)
    # print(predict[0,:,:,0])
    return predict

def embedding_set(model_dir,file_name):
    unet_model = load_model(model_dir,{'m_psnr':m_psnr})
    embedding = Model(inputs=unet_model.input, outputs = unet_model.get_layer("conv2d_5").output)
    embedding.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = [m_psnr])
    input_set = load_file(file_name)
    num_img = input_set.shape[0]
    # print(num_img)
    # print(input_set[0].shape)
    set = np.zeros((num_img,)+EMBEDDING_SHAPE,dtype=np.float32)
    # print(set[0].shape)
    # # print(set.shape)
    for i in range(num_img):
        em = embedding.predict(np.reshape(input_set[i],(1, IMG_H, IMG_W, NUM_C)))
        set[i] = em 
    save_name = "em_"+file_name.split(".npy")[-2]
    np.save(save_name,set)
    return set

    # for i in range(num)
    # img_list = glob.glob(img_dir+'*.png')
    # for img_name in img_list:
    #     img = np.array(Image.open(img_name))
    #     img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
    #     img = add_guassian(img)
    #     img = img.astype('float32') / 255.

if __name__ == "__main__":
    # embedding_set("./unet2_clean50-0.00.hdf5","clean24.npy")
    img = np.array(Image.open('./Kodak24_n/kodim01_1.png'))
    img = np.reshape(img, (1, IMG_H, IMG_W, NUM_C))
    unet_model = load_model("./unet2_clean50-0.00.hdf5",{'m_psnr':m_psnr})
    embedding_layer(unet_model,img)