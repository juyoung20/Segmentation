import random
import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
data_location = ''

training_images_loc = data_location + 'DRIVE/training/images/'
training_label_loc = data_location + 'DRIVE/training/1st_manual/'
mixup_images_loc = data_location + 'DRIVE/mixup/images'
mixup_labels_loc = data_location + 'DRIVE/mixup/labels'


train_files = os.listdir(training_images_loc)
train_data = []
train_label = []

desired_size = 592
for i in train_files:
    im = cv2.imread(training_images_loc + i)
    label = imageio.imread(training_label_loc + i.split('_')[0] + '_manual1.gif')
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)

train_data = np.array(train_data)
train_label = np.array(train_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), desired_size, desired_size, 3))
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))

def mixup(input1,input2, target1, target2, gamma = 0.05):
    input = input1.copy()
    target = target1.copy()
    h,w,c = input1.shape
    for i in range(c):
        for j in range(h):
            for k in range(w):
                    input[j][k][i] = input1[j][k][i]*gamma +  input2[j][k][i]* (1 - gamma)

    for j in range(h):
        for k in range(w):
            if k %2 == 0:                     
                target[j][k][0] = target1[j][k][0]*gamma +  target2[j][k][0]* (1 - gamma)
    return input, target

for i in range(100):
    r = random.randrange(0,20)
    r2 = random.randrange(0,20)
    print(r, r2)
    x_t, y_t = mixup(x_train[r], x_train[r2], y_train[r], y_train[r2])
    x_t = cv2.cvtColor((x_t*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    tmp_img_name = os.path.join(mixup_images_loc, str(i) + '_training.tif')
    imageio.imwrite(tmp_img_name, x_t)
 
    tmp_lab_name = os.path.join(mixup_labels_loc, str(i) + '_manual1.gif')
    y_t = cv2.cvtColor((y_t*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    imageio.imwrite(tmp_lab_name, y_t)

