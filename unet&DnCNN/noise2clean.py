from network import *
from embedding import *
from keras.callbacks import ModelCheckpoint

def noise2clean(n_em_dir,c_em_dir):
    n_data = np.load(n_em_dir)
    c_data = np.load(c_em_dir)
    mlp = MLP()
    mlp.compile(optimizer = Adam(lr = 1e-4), loss = 'mse')
    # mlp.summary()
    model_checkpoint = ModelCheckpoint('mlp_{epoch:02d}.hdf5', monitor='val_loss',verbose=1, save_best_only=True,mode='min')
    mlp.fit(n_data,c_data,BATCH_SIZE2,EPOCHS2,1,validation_split=0.3,callbacks=[model_checkpoint])

def denoise(img_name,n2c_model_name,c_model_name,n_model_name):
    img = load_img(img_name,True) #noisy image
    c_model = load_model(c_model_name)
    n_model = load_model(n_model_name)
    n2c_model = load_model(n2c_model_name)
    embedding = Model(inputs=n_model.input, outputs = n_model.get_layer("conv2d_5").output)
    decoding = Model(inputs=c_model.get_layer(""), outputs = n_model.output)
    


if __name__ == "__main__":
    n_dir = "./em_guassian24.npy"
    c_dir = "./em_clean24.npy"
    noise2clean(n_dir,c_dir)
