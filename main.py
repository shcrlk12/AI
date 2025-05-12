import tensorflow as tf
import numpy as np
import os
from numpy import expand_dims
from matplotlib import pyplot as plt
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import img_to_array

img = load_img('test.png')
data = img_to_array(img)

samples = expand_dims(data, 0)

datagen = ImageDataGenerator(horizontal_flip=True)
it = datagen.flow(
                    samples, batch_size=1, 
                    save_to_dir='aug_data',  # 여기에 저장됨
                    save_prefix='aug',
                    save_format='png'  # 또는 'jpg'
                )
fig = plt.figure(figsize=(30,30))

for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = it.__next__()
    image = batch[0].astype('uint8')
    plt.imshow(image)

plt.show()