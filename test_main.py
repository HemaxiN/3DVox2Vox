from test import *

models_path = r'/dev/shm/dataset3d/model_033000.h5'
input_path = r'/dev/shm/dataset3d/test/masks'
save_dir = r'/dev/shm/dataset3d/syntheticimages'

image_shape = (64,256,256,3)

predict(models_path, input_path, save_dir, image_shape)