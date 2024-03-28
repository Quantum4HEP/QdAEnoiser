import numpy as np

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

import os


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
seed = 123456
tf.random.set_seed(seed)



def data_job(index):
    np.random.seed(index[0])
    expscale = index[1]
    normloc = index[2]
    bins = index[3]
    distr = index[4]
    if distr == 0:
        cleandata = np.random.normal(loc=normloc,size = 5000)
    else:
        cleandata = np.random.exponential(scale=expscale,size = 5000)
    cleandata = np.concatenate((cleandata,np.random.normal(loc=4, scale = 2, size = 5000)))
    cleanhist = np.histogram(cleandata,  bins = np.linspace(bins[0],bins[1],1025), density = True)[0]
    
    noise = np.random.normal(size = 10000)
    
    dirtydata = cleandata + noise + 0.6 # + np.random.uniform(low=-10, high = 10,size=10000)
    dirtydata = np.concatenate((dirtydata,np.random.uniform(low=-10, high = 10,size=10000)))
    dirtyhist = np.histogram(dirtydata,  bins = np.linspace(bins[0],bins[1],1025), density = True)[0]

    return cleanhist, dirtyhist


def generate_data(n):
    expscale = [] 
    normloc = []
    bins = []
    distr = []
    for i in range(n):
        expscale.append(1.0)
        normloc.append(-4)
        bins.append([-10,10])
        distr.append(0)
    with multiprocessing.Pool(processes=11) as pool:
        results = pool.map(data_job, zip(range(n), expscale, normloc, bins,distr))
    results = np.array(results)
    print(results.shape)
    dirtypath = f"dirtyset_double_biased_1024_{n}"
    cleanpath = f"cleanset_double_biased_1024_{n}"
    np.save(dirtypath, results[:,1])
    np.save(cleanpath, results[:,0])
    return dirtypath + ".npy",cleanpath +".npy"





def kl_loss(y_true, y_pred):
    # Ensure that both distributions sum up to 1
    y_true = tf.keras.backend.clip(y_true, 1e-10, 1)
    y_pred = tf.keras.backend.clip(y_pred, 1e-10, 1)
    
    # Compute KL divergence
    kl_loss = tf.reduce_sum(y_true * tf.math.abs(tf.math.log(y_true / y_pred)), axis=-1)
    return kl_loss

def mse_loss(y_true, y_pred):
    se = tf.math.square(y_true - y_pred)
    return tf.math.reduce_mean(se)


def custom_loss(y_true,y_pred):
    return mse_loss(y_true,y_pred) + 0.05*kl_loss(y_true,y_pred)




def Denoiser(input_shape, latent_dim, pruning='not_pruned', expose_latent=False):

    # encoder
    input_encoder = tf.keras.layers.Input(shape=input_shape, name='encoder_input')

    # x = tf.keras.layers.Dense(64,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(input_encoder)    
    # x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Reshape((32,32,1))(input_encoder)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2,2), use_bias=True, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(2,2), use_bias=False, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(2,2), use_bias=False, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=(2,2), use_bias=False, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.MaxPooling2D((2,2))
    # x = tf.keras.layers.Dense(30,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    #     
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(20,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    # x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed))(x)
    x = tf.keras.layers.Activation('relu')(x)


    
    enc = tf.keras.layers.Dense(latent_dim,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed))(x)

    encoder = tf.keras.models.Model(inputs=input_encoder, outputs=enc)
    encoder.summary()
    # decoder
    input_decoder = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    x = tf.keras.layers.Dense(256,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed))(input_decoder) 
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Dense(512,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed))(x) 
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Reshape((4,4,32))(x)

    #x = tf.keras.layers.UpSampling2D((4,4))(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(2,2), use_bias=False, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.UpSampling2D((2,2))(x)

    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(2,2), use_bias=False, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.UpSampling2D((2,2))(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(2,2), use_bias=True, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.UpSampling2D((2,2))(x)

    x = tf.keras.layers.Conv2DTranspose(1, kernel_size=(2,2), use_bias=False, padding='same', kernel_initializer = tf.keras.initializers.GlorotNormal(seed=seed))(x)
    x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.UpSampling2D((2,2))(x)

    dec = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x) 
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x) 
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x) 
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(20,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x) 
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(30,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x) 
    # x = tf.keras.layers.Activation('relu')(x)

    # x = tf.keras.layers.Dense(64,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed))(x) 
    # x = tf.keras.layers.Activation('relu')(x) 


    # dec = tf.keras.layers.Dense(input_shape,kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=seed),
    #                              kernel_regularizer=tf.keras.regularizers.L2(0.0001),activation='relu')(x)
    #
    decoder = tf.keras.models.Model(inputs=input_decoder, outputs=dec)
    decoder.summary()

    # if pruning=='pruned':
    #     start_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 5
    #     end_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 15
    #     pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    #                             initial_sparsity=0.0, final_sparsity=0.5,
    #                             begin_step=start_pruning, end_step=end_pruning)
    #     encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
    #     encoder = encoder_pruned
    #     decoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(decoder, pruning_schedule=pruning_schedule)
    #     decoder = decoder_pruned

    # ae
    if expose_latent:
        latent = encoder(input_encoder)
        ae_outputs = decoder(latent)
        autoencoder = tf.keras.models.Model(inputs=input_encoder, outputs=(latent, ae_outputs))
    else:
        ae_outputs = decoder(encoder(input_encoder))
        autoencoder = tf.keras.models.Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()
    # load weights
    # if pruning=='pruned':
    #     autoencoder = model_set_weights(autoencoder, f'output/model-conv_ae-8-b0-q0-not_pruned', quant_size)
    # compile AE

    
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=3E-2, amsgrad=True),
        loss=custom_loss)
    return autoencoder







def run(dirtypath,cleanpath):

    dirtyset = np.load(dirtypath) # "dirtyset_double_biased_10000.npy"
    cleanset = np.load(cleanpath) # "cleanset_double_biased_10000.npy"

    dirt_train, dirt_val, clean_train, clean_val = train_test_split(dirtyset,cleanset, test_size=0.2, random_state=seed) # scaler.fit_transform(dirtyset)
    # scaler.fit_transform(cleanset)
    model = Denoiser(1024,128)
    
    test = np.load("dirtyset_double_biased_1024_test_1000.npy")
    cleantest = np.load("cleanset_double_biased_1024_test_1000.npy")
    testexpo = np.load("dirtyset_expo_biased_1024_test_1000.npy")
    cleanexpo = np.load("cleanset_expo_biased_1024_test_1000.npy")
    callbacks=[
        # TensorBoard(log_dir="tb_logs", histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=6, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-9)
        ]
    history = model.fit(x = dirt_train, y = clean_train, epochs= 100, batch_size = 2048, validation_data = (dirt_val, clean_val), callbacks = callbacks)
    print("####################################")
    print("Validation loss last epoch: ", history.history['val_loss'][-1])
    y = model.predict(test)
    yexpo = model.predict(testexpo)
    model.save("denoiser.h5")
    print("TEST + NOISE ", test)
    sample = np.random.randint(0,1000)
    print(y[sample])
    print("CLEAN TEST ", cleantest[sample])
    plt.plot(range(1024),test[sample], label='Noisy')
    plt.plot(range(1024),cleantest[sample],label='Clean')
    plt.plot(range(1024),y[sample], label='Pred')
    plt.legend()
    plt.text(800,0.15, "Val_loss " + str(history.history['val_loss'][-1]))
    plt.savefig("quicktestnorm.png")
    plt.clf()
    plt.plot(range(1024),testexpo[sample], label='Noisy')
    plt.plot(range(1024),cleanexpo[sample],label='Clean')
    plt.plot(range(1024),yexpo[sample], label='Pred')
    plt.legend()
    plt.text(800,0.15, "Val_loss " + str(history.history['val_loss'][-1]))
    plt.savefig("quicktestexpo.png")


def noisetest(dirtypath,cleanpath):

    dirtyset = np.load(dirtypath)
    cleanset = np.load(cleanpath)

    se = np.square(cleanset - dirtyset)
    # print(se.shape)
    mse = np.mean(se,axis=1)
    # print(mse.shape)
    
    
    sample = np.random.randint(0,len(cleanset))
    print("Plotting random event: idx: ",sample)
    plt.plot(dirtyset[sample], label='Noisy')
    plt.plot(cleanset[sample], label= 'Clean')
    plt.legend()
    plt.show()


def mixgenerator(n):
    expscale = [] 
    normloc = []
    bins = []
    distr = []
    for i in range(n):
        expscale.append(np.random.uniform(0.5,5))
        normloc.append(np.random.uniform(-8.0,8.0))
        distr.append(np.random.randint(0,2))
        if distr[-1] == 0:
            bins.append([-10,10])
        else:
            bins.append([-1,10])

    with multiprocessing.Pool(processes=11) as pool:
        results = pool.map(data_job, zip(range(n), expscale, normloc, bins,distr))
    results = np.array(results)
    print(results.shape)
    dirtypath = f"dirtyset_mixed_{n}"
    cleanpath = f"cleanset_mixed_{n}"
    np.save(dirtypath, results[:,1])
    np.save(cleanpath, results[:,0])
    return dirtypath + ".npy",cleanpath +".npy"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, help='Generate data', required=False, default = 0)
    parser.add_argument('-i', type=str, nargs='+', help='Input files (noisy data first)', required=False, default = None)
    parser.add_argument('--mixgen', type=int, help='Generate mixed data', required=False, default = 0)
    args = parser.parse_args()
    if (args.g != 0):
        print("Generating dirty and clean data")
        d,c = generate_data(args.g)
        print("Data generated saved in ", d,c)
        noisetest(d,c)
    if (args.i != None):
        run(args.i[0],args.i[1])
    if (args.mixgen != 0):
        print("Generating dirty and clean data of random distrs")
        d,c = mixgenerator(args.mixgen)
        print("Data generated saved in ", d,c)
        noisetest(d,c)
    if (args.g == 0 and args.i == None and args.mixgen == 0):
        print("\n\n")
        parser.print_help()
