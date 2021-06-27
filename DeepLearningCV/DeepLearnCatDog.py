import cv2
import matplotlib.pyplot as plt

# CAT_DOGS DATASET LINK => https://www.kaggle.com/tongpython/cat-and-dog

# We show sample image from dataset with matplotlib.
cat = cv2.imread('CATS_DOGS/train/CAT/4.jpg')
cat = cv2.cvtColor(cat,cv2.COLOR_BGR2RGB)
# plt.imshow(cat)
# plt.show()

dog = cv2.imread('CATS_DOGS/train/DOG/2.jpg')
dog = cv2.cvtColor(dog,cv2.COLOR_BGR2RGB)
# plt.imshow(dog)
# plt.show()


from keras.preprocessing.image import ImageDataGenerator

# In this section, we can obtain more images by making minor changes on the image data we have.
image_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              rescale=1/255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')


# Examples of images obtained using ImageDataGenerator.
plt.imshow(image_gen.random_transform(dog))
plt.show()

# Finds the total number of files under the file and tells how many classes it consists of.
image_gen.flow_from_directory('CATS_DOGS/train')


from keras.models import Sequential

# The part where we build our model.

model = Sequential()

"""
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


model.summary()
"""

from keras.preprocessing import image

batch_size = 16
input_shape = (150,150,3)

# Takes the path to a directory & generates batches of augmented data.

train_image_gen = image_gen.flow_from_directory('CATS_DOGS/train',
                                               target_size=input_shape[:2],
                                               batch_size = batch_size,
                                               class_mode='binary')

test_image_gen = image_gen.flow_from_directory('CATS_DOGS/test',
                                               target_size=input_shape[:2],
                                               batch_size = batch_size,
                                               class_mode='binary')

# Tells how many classes it consists of.
print(train_image_gen.class_indices)

"""
# Trains the model for a fixed number of epochs (iterations on a dataset)
results = model.fit(train_image_gen,epochs=1,steps_per_epoch=150,
                             validation_data=test_image_gen,validation_steps=12)

# Returns the model's display labels for accuracy outputs.
print(results.history['accuracy'])
"""


from keras.models import load_model

# It is used to load ready models.
new_model = load_model('cat_dog_100epochs.h5')

# We are making a test image input to test the model.
dog_file = 'CATS_DOGS/test/DOG/10004.jpg'
dog_img = image.load_img(dog_file,target_size=(150,150))
dog_img = image.img_to_array(dog_img)


import numpy as np

# We control the prediction processes to the model. ( For Single Data )
dog_img = np.expand_dims(dog_img,axis=0)
# Scaled image.
dog_img = dog_img/255
# The model predicts incoming input.
print(new_model.predict_classes(dog_img))
print(new_model.predict(dog_img))