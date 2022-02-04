import os
import pdb
import numpy as np

import numpy as np

label_list = []
sixteenozcan = []
twentyozbottle = []
iso = []
slimcan = []
label = '/multiview/3d-count/obj_detection/normalized_data/training/label_2/'
for filename in os.listdir(label):
    f = open(label + filename)
    for line in f:
        label_list.append(line.split())
for i in label_list:
    if i[0] == '16ozcan':
        sixteenozcan.append(i)
    elif i[0] == '2Oozbottle':
        twentyozbottle.append(i)
    elif i[0] == '16ozIsotonicbottle':
        iso.append(i)
    elif i[0] == '12ozSlimcan':
        slimcan.append(i)
    # elif i[0] == 'sixpack':
    #     sixpack.append(i)




class_list = [sixteenozcan,twentyozbottle, iso, slimcan]
class_list_averages = []
for class_name in class_list:
    h = []
    w = []
    l = []
    x = []
    y = []
    z = []
    averages = []
    for i in class_name:
        h.append(float(i[8]))
        w.append(float(i[9]))
        l.append(float(i[10]))
        x.append(float(i[11]))
        y.append(float(i[12]))
        z.append(float(i[13]))
    averages.append(np.average(h))
    averages.append(np.average(w))
    averages.append(np.average(l))
    averages.append(np.average(x))
    averages.append(np.average(y))
    averages.append(np.average(z))
    class_list_averages.append(averages)
print(class_list_averages)



