import sys
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import pathlib
import time
import matplotlib.pyplot as plt
from keras.src.layers.activations import activation

start_time = time.time()
# point to the url of the data
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# extract=true indicates if the downloaded file is in zip form to get extracted
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)

# creates a path to the archive and removes the file extension from the path
data_dir = pathlib.Path(archive).with_suffix('')

# lists all the items with the .jpg extension from data dir and puts them in list to count
#image_count = len(list(data_dir.glob('*/*.jpg')))

#print(image_count)

'''
roses = list(data_dir.glob('roses/*'))
image_path = str(roses[5])
image = Image.open(image_path)
image.show()
'''

'''Creating a Dataset'''
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # spares 20% of the data for validation instead of training
    subset="training",  # specifies the type of data utilized in this case the training
    seed=123,  # the randomizerz
    image_size=(img_height, img_width),
    batch_size=batch_size
)
# gets 20% of the whole dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

#class_names = train_ds.class_names
#print(class_names)

'''Normalize the data'''


# lambda x, y: (normalization_layer(x), y) is exactly the same
def norm(x, y):
    return normalization_layer(x), y


# standardization layer converting the RGB values to be in the range of [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

# normalize the training data, lambda is a short function type in python
normalized_ds = train_ds.map(norm)

# takes the images from normalized data distributes them to new batches
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

'''
# np.min and np.max finds the lowest and the biggest values in the array
print(np.min(first_image), np.max(first_image))
'''
AUTOTUNE = tf.data.AUTOTUNE
# store the data in cache and prefetching
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

# convolutional neural network (CNN)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)

])

'''Conv2D applies 32 filters/kernels to the input,
    each filter/kernel is the size of 3*3 the second argument,
    ReLu stand for Rectified Linear Unit
'''

'''
MaxPooling is a downSampling operation
It reduces the spatial dimension of the data
'''


model.compile(
    optimizer= 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time is: ", elapsed_time, "seconds")
