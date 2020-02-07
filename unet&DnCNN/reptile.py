from network import *
from utils import *
import random
from keras.callbacks import ModelCheckpoint

def weight_minus(weight_a,weight_b):
    weight_delta = []
    lenth = len(weight_a)
    for i in range(lenth):
        weight_delta.append(weight_a[i]-weight_b[i])
    # print(weight_delta[0].shape)
    return weight_delta


def reptile(img_set_name):
    img_set = load_file(img_set_name)
    # print(img_set.shape)
    model = Unet()
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = [m_psnr])
    weight = model.get_weights()
    # print(weight[0].shape)
    # print(np.array(weight).shape)
    for epoch in range(EPOCHS_REPTILE_OUT):
        # print("Outer epoch: " + str(epoch))
        delta_weights = []
        for w in weight:
            delta_weights.append(np.zeros_like(w))
        model.set_weights(weight)
        for task in range(TASK_REPTILE_NUM):
            guassian_type = GUASSIAN_TYPE[random.randint(0,len(GUASSIAN_TYPE)-1)]
            print("Outer epoch: " + str(epoch) + "\tTask_num: " + str(task) + "\tsigma: " + str(guassian_type))
            sample_set = np.random.permutation(img_set)[:SAMPLE_REPTILE_NUM]
            g_samle_set = add_guassian(sample_set,sigma=guassian_type,normalized=True)
            model.fit(g_samle_set,sample_set,BATCH_REPTILE_SIZE,EPOCHS_REPTILE_IN)
            new_weight = model.get_weights()
            delta_weights += weight_minus(new_weight,weight)
        for i in range(len(weight)):
            weight[i] = weight[i] + REPTILE_EPSILON/TASK_REPTILE_NUM*delta_weights[i]
    final_weight = model.get_weights()
    model.set_weights(weight)
    model.save_weights("./reptile%d-%d-%d-%d.hdf5" % (EPOCHS_REPTILE_OUT,TASK_REPTILE_NUM,SAMPLE_REPTILE_NUM,EPOCHS_REPTILE_IN))
    return final_weight
            
def train_with_reptile(phi):
    model = Unet()
    model.load_weights("./reptile5-3-200-20.hdf5")
    # model.set_weights(phi)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = [m_psnr])
    original_img = load_file("./clean-sigma25_6000.npy")
    noise_img = load_file("./guassian-sigma25_6000.npy")
    model_checkpoint = ModelCheckpoint('reptile_clean{epoch:02d}.hdf5', monitor='val_loss',verbose=1, save_best_only=True,mode='min')
    model.fit(noise_img,original_img,BATCH_SIZE,EPOCHS,1,validation_split=0.3,callbacks=[model_checkpoint])

if __name__ == "__main__":
    phi = reptile("./clean24.npy")
    train_with_reptile(phi)

