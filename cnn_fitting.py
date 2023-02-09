#Setting keras seeds: https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
from keras.models import Sequential
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from tkinter import filedialog
from tkinter import *
from tkinter.filedialog import askopenfilename
import os

from PIL import Image, ImageOps

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


import csv

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU

from sklearn.metrics import confusion_matrix

IMAGE_SIZE = 50,50

POKEMON_LABELS = {
    0: "Bulbasaur",
    1: "Charmander",
    2: "Mew",
    3: "Pikachu",
    4: "Squirtle"
}
#Function to read and format data from csv files
def get_data(filename):
    print("Collecting Image Data..")
    data = []

    with open(filename) as data_file:
        reader = csv.reader(data_file, delimiter=',')
        for row in reader:
            data.append(row)

    data.pop(0)

    data_X_flat = []
    data_Y = []

    print("Flattening Images")

    for i in range(0, len(data)):
        data_Y.append(data[i][-1])
        data[i].pop(len(data[i])-1)
        data_X_flat.append(data[i])


    print("Coverting Images to Matrix")

    data_X = []
    #Convert from float
    for img_flat in data_X_flat:
        image = []
        for i in range(0, 50):
            x1 = i*50
            x2 = i*50 + 50
            layer = img_flat[x1:x2]
            image.append(layer)
        data_X.append(image)

    o1 = np.asarray(data_X).astype(np.float32)
    o2 = np.asarray(data_Y).astype(np.int)

    return (o1, o2)

def convert_image(im):
    greyscale_flat = []
    X = []

    greyscale = ImageOps.grayscale(im)
    greyscale_resized = greyscale.resize(IMAGE_SIZE, Image.ANTIALIAS)
    #greyscale_resized.save(folder_selected + "/downscaled/" + i, dpi=(50,50))

    pix = greyscale_resized.load()
    pixels = []
    for x in range(0, 50):
        for y in range(0, 50):
            pixels.append(pix[x, y]/255)

    greyscale_flat.append(pixels)

    image = []
    for i in range(0, 50):
        x1 = i*50
        x2 = i*50 + 50
        layer = greyscale_flat[0][x1:x2]
        image.append(layer)
    X.append(image)


    return X


#Shape 50x50x1

(test_X, test_Y) = get_data("test_data.csv")
(train_X, train_Y) = get_data("training_data.csv")

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

batch_size = 64
epochs = 16
num_classes = 5

pokemon_model = Sequential()
pokemon_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(50,50,1),padding='same'))
pokemon_model.add(LeakyReLU(alpha=0.1))
pokemon_model.add(MaxPooling2D((2, 2),padding='same'))
pokemon_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
pokemon_model.add(LeakyReLU(alpha=0.1))
pokemon_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
pokemon_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
pokemon_model.add(LeakyReLU(alpha=0.1))                  
pokemon_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
pokemon_model.add(Flatten())
pokemon_model.add(Dense(128, activation='linear'))
pokemon_model.add(LeakyReLU(alpha=0.1))                  

pokemon_model.add(Dropout(0.3))

pokemon_model.add(Dense(num_classes, activation="softmax"))

pokemon_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#pokemon_model.summary()

#pokemon_train = pokemon_model.fit(train_X, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1)
pokemon_train = pokemon_model.fit(train_X, train_Y_one_hot, batch_size=batch_size,epochs=epochs,
    verbose=1,validation_data=(test_X, test_Y_one_hot))

from sklearn.metrics import confusion_matrix

#Predict
y_prediction = pokemon_model.predict(test_X)
predicted_classes = np.argmax(np.round(y_prediction),axis=1)

#Create confusion matrix and normalizes it over predicted (columns)
cm = confusion_matrix(test_Y, predicted_classes, normalize='pred')
print(cm)

df_cm = pd.DataFrame(cm, index = [i for i in "BCMPS"],
                  columns = [i for i in "BCMPS"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

plt.plot(pokemon_train.history['accuracy'], label='Training Accuracy')
plt.plot(pokemon_train.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

plt.show()

plt.plot(pokemon_train.history['loss'], label='Training Loss')
plt.plot(pokemon_train.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, 2])
plt.legend(loc='lower right')

plt.show()

test_eval = pokemon_model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

while(input("Enter Image to Test? (Y/N): ") == "Y"):
    print("Asking for Test Image...")

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    image_filepath = filedialog.askopenfilename()
    im = Image.open(image_filepath)

    test_image = convert_image(im)


    predicted_classes = pokemon_model.predict(test_image)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

    label = predicted_classes[0]

    print("Image Recognised as " + str(label) + ": " + POKEMON_LABELS[label])
