from keras.datasets import fashion_mnist

# We load the dataset from keras with fashion_mnist library.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt

# We show sample image from dataset with matplotlib.
plt.imshow(x_train[0])
plt.show()

# Scale dataset in the range 0 to 1.
x_train = x_train/255
x_test = x_test/255

# Reshape the inputs in the dataset for the input shape of the model.
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

from keras.utils.np_utils import to_categorical

# We used the one hot encoding method to classify more healthily.
y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

# The part where we build our model.
model = Sequential()

model.add(Conv2D(filters = 32 ,kernel_size = (4,4) , input_shape = (28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

# Summary table of the model we created.
model.summary()

# Trains the model for a fixed number of epochs (iterations on a dataset)
model.fit(x_train,y_cat_train,epochs=5)

# Returns the model's display labels for all outputs.
print(model.metrics_names)

# Trains the model. (Computation is done in batches)
model.evaluate(x_test,y_cat_test)

# We control the prediction processes to the model.
predic = model.predict_classes(x_test)

print(predic)
print(y_test)

from sklearn.metrics import classification_report

# We build a text report showed the main classification metrics.
# Parameters of table as precision,recall,f1-score and support.
print(classification_report(y_test,predic))

import numpy as np

# We control the prediction processes to the model. ( For Single Data )
x_test_new = np.expand_dims(x_test[0],axis=0)
print(model.predict_classes(x_test_new))
print(y_test[0])
