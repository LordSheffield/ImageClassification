import numpy as np
import tensorflow as tf
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model(os.path.join('models', 'elderscrollsmodel.h5'))

img = cv2.imread('Morrowind.jpg')
resize = tf.image.resize(img, (256,256))
what = np.argmax(model.predict(np.expand_dims(resize/255, 0)))

if (what == 0):
    print("That screenshot is from The Elder Scrolls III: Morrowind.")

elif (what == 1):
    print("That screenshot is from The Elder Scrolls IV: Oblivion.")

elif (what == 2):
    print("That screenshot is from The Elder Scrolls V: Skyrim.")