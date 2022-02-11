import os
import pdb
import numpy as np
import tqdm
import numpy as np

cloud = '/multiview/3d-count/obj_detection/normalized_data/testing/velodyne/'
x = []
y = []
z = []
counter = 0
for filename in os.listdir(cloud):
    pc = np.fromfile(str(cloud + filename), dtype=np.float32).reshape(-1, 4)[:,0:-1]
    x.extend(pc[:, 0])
    y.extend(pc[:, 1])
    z.extend(pc[:, 2])
    counter = counter + 1
# pdb.set_trace()
x_min = np.min(x)
y_min = np.min(y)
z_min = np.min(z)

x_max = np.max(x)
y_max = np.max(y)
z_max = np.max(z)

print("x_min: ", x_min)
print("y_min: ", y_min)
print("z_min: ", z_min)

print("x_max: ", x_max)
print("y_max: ", y_max)
print("z_max: ", z_max)
