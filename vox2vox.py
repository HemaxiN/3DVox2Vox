# example of Vox2Vox GAN for image-to-image translation, code based on:
#https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import tifffile
import os

# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv3D(32, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv3D(256, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv3D(256, (4,4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv3D(n_filters, (4,4,4), strides=(1,2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv3DTranspose(n_filters, (4,4,4), strides=(1,2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape=(64,256,256,3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 32, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128)
    e4 = define_encoder_block(e3, 256)
    e5 = define_encoder_block(e4, 256)
    e6 = define_encoder_block(e5, 256)
    e7 = define_encoder_block(e6, 256)
    # bottleneck, no batch norm and relu
    b = Conv3D(512, (4,4,4), strides=(1,2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 256)
    d2 = decoder_block(d1, e6, 256)
    d3 = decoder_block(d2, e5, 256)
    d4 = decoder_block(d3, e4, 256, dropout=False)
    d5 = decoder_block(d4, e3, 128, dropout=False)
    d6 = decoder_block(d5, e2, 64, dropout=False)
    d7 = decoder_block(d6, e1, 32, dropout=False)
    # output
    g = Conv3DTranspose(3, (4,4,4), strides=(1,2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
            
    opt = Adam(lr=0.0002, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])            
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model
    

def load_real_samples(ix, dir_, img_shape):
    X1=zeros((len(ix),img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    X2=zeros((len(ix),img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    k=0
    for i in ix:
        image = tifffile.imread(os.path.join(dir_, 'images/'+str(i)+'.tif')) # RGB image
        mask = tifffile.imread(os.path.join(dir_, 'masks/'+str(i)+'.tif')) # RGB image
        X1[k,:]=(image-127.5) /127.5
        X2[k,:]=(mask-127.5) /127.5
        k=k+1
    return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(n_patches, dir_, img_shape, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, n_patches, n_samples)
    # retrieve selected images
    X1, X2 = load_real_samples(ix, dir_, img_shape)
    #X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 4, patch_shape, patch_shape, 1))
    return [X1, X2], y

# select a batch of random samples, returns images and target
def generate_real_samples2(ix, n_samples, patch_shape, dir_, img_shape):
    # retrieve selected images
    X1, X2 = load_real_samples(ix, dir_, img_shape)
    # generate 'real' class labels (1)
    y = ones((n_samples, 4, patch_shape, patch_shape, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), 4, patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, n_patches_val, val_dir, img_shape, n_samples=3):
    # select a sample of input images
    [X_realB, X_realA], _ = generate_real_samples(n_patches_val, val_dir, img_shape, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    
   
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i,31])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i,31])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i,31])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

    for i in range(n_samples):
        #ii = randrange(len(X_realA))
        filename3 = 'realimg_%06d_%03d.tif' % (step+1, i)
        save_img = X_realA[i]*255.0
        save_img = save_img.astype('uint8')
        tifffile.imsave(filename3, save_img, photometric='rgb')
        filename4 = 'fakeimg_%06d_%03d.tif' % (step+1, i)
        save_img = X_fakeB[i]*255.0
        save_img = save_img.astype('uint8')
        tifffile.imsave(filename4, save_img, photometric='rgb')
        filename4 = 'realmsk_%06d_%03d.tif' % (step+1, i)
        save_img = X_realB[i]*255.0
        save_img = save_img.astype('uint8')
        tifffile.imsave(filename4, save_img, photometric='rgb')



# train Vox2Vox model
def train(d_model, g_model, gan_model, img_shape, train_dir, val_dir, n_epochs=200, n_batch=1):    
    # validation patches and training patches
    n_patches_val = len(os.listdir(os.path.join(val_dir, 'images/')))
    n_patches_train = len(os.listdir(os.path.join(train_dir, 'images/')))
    print('Number of Training Samples: %i' % n_patches_train)
    print('Number of Validation Samples: %i' % n_patches_val)	

    # determine the output square shape of the discriminator
    #save the losses
    losses_list = []
    n_patch = d_model.output_shape[2]
    # unpack dataset
    bat_per_epo = int(n_patches_train / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    #steps counter
    i = 0
    # manually enumerate epochs
    for k in range(n_epochs):
        # select a batch of real samples
        array_samples = np.arange(0,n_patches_train)
        np.random.shuffle(array_samples)
        for sample_ in array_samples:
            [X_realB, X_realA], y_real = generate_real_samples2([sample_], n_batch, n_patch, train_dir, img_shape)
            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            losses_list.append([i+1, d_loss1, d_loss2, g_loss])
            # summarize model performance
            if (i+1) % (bat_per_epo * 10) == 0:
                summarize_performance(i, g_model, n_patches_val, val_dir, img_shape)
            i = i+1 #steps
    np.save('listlosses.npy',losses_list)