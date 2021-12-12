from vox2vox import *

train_dir = r'/dev/shm/dataset3d/train' #directory with the training set
val_dir = r'/dev/shm/dataset3d/val' #directory with the validation set

image_shape = (64, 256, 256, 3)

# define the discriminator and generator models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train the model
train(d_model, g_model, gan_model, image_shape, train_dir, val_dir, n_epochs=200, n_batch=1)
