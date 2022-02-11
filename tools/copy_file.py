import shutil
import os
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt


large_image = '/multiview/3d-count/obj_detection/normalized_data/training/000000.png'
calib_file = '/multiview/3d-count/obj_detection/normalized_data/training/000000.txt'
image_file = '/multiview/3d-count/obj_detection/normalized_data/training/small_image.png'
im = plt.imread(large_image)
res = resize(im, (612, 185))
plt.imsave(image_file,res)


for i in tqdm(range(0,128506)):
    hi = str(i)
    new_file = '/multiview/3d-count/obj_detection/normalized_data/testing/image_2/' + str(hi.zfill(6)) + '.png'
    shutil.copy(image_file, new_file)
    

for i in tqdm(range(0,128506)):
    hi = str(i)
    new_file = '/multiview/3d-count/obj_detection/normalized_data/testing/calib/' + str(hi.zfill(6)) + '.txt'
    shutil.copy(calib_file, new_file)

for i in tqdm(range(0,503063)):
    hi = str(i)
    new_file = '/multiview/3d-count/obj_detection/normalized_data/training/image_2/' + str(hi.zfill(6)) + '.png'
    shutil.copy(image_file, new_file)

for i in tqdm(range(0,503063)):
    hi = str(i)
    new_file = '/multiview/3d-count/obj_detection/normalized_data/training/calib/' + str(hi.zfill(6)) + '.txt'
    shutil.copy(calib_file, new_file)