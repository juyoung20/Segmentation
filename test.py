import os
import cv2
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from model import *

data_location = ''

test_images_loc = data_location + 'DRIVE/test/images/'
test_label_loc = data_location + 'DRIVE/test/1st_manual/'


test_files = os.listdir(test_images_loc)
test_data = []
test_label = []

desired_size = 592
for i in test_files:
    im = cv2.imread(test_images_loc + i)
    label = cv2.imread(test_label_loc + i.split('_')[0] + '_manual1.gif')
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

    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)

test_data = np.array(test_data)
test_label = np.array(test_label)


x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_test= np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

if os.path.exists('./save_weights/unet.h5'):
    model = UNET.load_weights("./save_weights/unet.h5")
test = model.evaluate(x_test, y_test, batch_size = 4)
