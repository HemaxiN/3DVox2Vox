from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from numpy import zeros
import tifffile
import numpy as np
import os


def load_real_samples(input_path, ix, img_shape):
    X2=zeros((len(ix),img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    k=0
    for i in ix:
        mask = tifffile.imread(os.path.join(input_path , str(i)+'.tif')) # RGB image
        X2[k,:]=(mask-127.5) /127.5
        k=k+1
    return X2

# generate samples and save as a plot and save the model
def summarize_performance(g_model, samples_, input_path, save_dir, img_shape):
    for s in samples_:
        # select a sample of input images
        X_realA = load_real_samples(input_path, [s], img_shape)
        # generate a batch of fake samples
        X_fakeB = g_model.predict(X_realA)
        # scale all pixels from [-1,1] to [0,1]
        X_fakeB = (X_fakeB + 1) / 2.0
        filename = os.path.join(save_dir,'%03d.tif' % (s))
        save_img = X_fakeB[0]*255.0
        save_img = save_img.astype('uint8')
        tifffile.imsave(filename, save_img, photometric='rgb')

def predict(models_path, input_path, save_dir, img_shape):
    patch_numbers = len(os.listdir(input_path))
    print("Number of Test Samples: %i" % patch_numbers)
    g_model = load_model(models_path)
    summarize_performance(g_model, patch_numbers, input_path, save_dir, img_shape)

