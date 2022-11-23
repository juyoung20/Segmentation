import os
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import jaccard_score, accuracy_score,classification_report

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
    label = imageio.v2.imread(test_label_loc + i.split('_')[0] + '_manual1.gif')
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

#test = model.evaluate(x_test, y_test, batch_size = 4)

class UNET_test(object):
    def __init__(self):
        # create unet
        n_classes = 1
        self.unet = UNET(n_classes)

        # compile
        lr = 1e-3
        self.unet.compile(optimizer = tf.keras.optimizers.Adam(lr = lr), loss = 'binary_crossentropy', metrics = ['accuracy'])

    def eval(self, x_test, y_test):
        if os.path.exists('unet_binary.h5'):
            self.unet.build(input_shape=(None, 592, 592, 3))
            self.unet.load_weights('unet_binary.h5')
        else:
            return 0

        pred_seg = self.unet(x_test)

        plt.figure(figsize=(20, 6))

        n = 5
        ini = random.randint(0, 20-n)

        for i in range(n):
            # display image
            
            ax = plt.subplot(3, n, i + 1)
            image = cv2.cvtColor(x_test[ini+i], cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title(str(ini+i) + "original")
            plt.savefig('original.png')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display true segmentation
            ax = plt.subplot(3, n, i + 1 + n)
            image = cv2.cvtColor((y_test[ini+i]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title(str(ini+i) + "true segmentation")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display predictions
            ax = plt.subplot(3, n, i + 1 + n + n)
            pred = pred_seg[ini + i].numpy()
            pred[pred > 0.5]  = 1
            pred[pred <= 0.5] = 0
            image = cv2.cvtColor((pred*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title("prediction")
            plt.savefig('prediction.png')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

a = UNET_test()
a.eval(x_test, y_test)
