from evaluation import *

img1_dir = '/dev/shm/dataset3d/test/images' #test images
img2_dir = '/dev/shm/dataset3d/test/syntheticimages' #synthetic images
compute_FID(img1_dir, img2_dir)