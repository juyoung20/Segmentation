import os
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from model import *

data_location = ''

training_images_loc = data_location + 'DRIVE/mixup/images/'
training_label_loc = data_location + 'DRIVE/mixup/labels/'


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

model = UNET(n_classes = 1)
lr = 1e-3

model.compile(optimizer = tf.keras.optimizers.Adam(lr = lr), loss ='binary_crossentropy', metrics = ['accuracy'])
#model.build(input_shape = (None, 592, 592, 3))
#model.summary()

history=model.fit(x_train, y_train, epochs=200, batch_size=20,
                validation_split=0.2,shuffle=True,verbose = 2)

model.save_weights(filepath = 'unet_mixup.h5')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('UNet Accuracy - mixup')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='lower right')
plt.savefig('accuracy_mixup.png')
plt.show()
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('UNet loss - mixup')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc= 'upper right')
plt.savefig('loss_mixup.png')
plt.show()
