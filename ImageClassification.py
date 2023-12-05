import tensorflow as tf
import os
import cv2
import imghdr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Image quality check.
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir('data'):
    for image in os.listdir(os.path.join('data', image_class)):
        image_path = os.path.join('data', image_class, image)
        try:
            img = cv2.imread(image_path)
            tip= imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in exts list {}'.format(image.path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

data = tf.keras.utils.image_dataset_from_directory('data', label_mode='categorical')
scaled_data = data.map(lambda x,y: (x/255, y))

#Allocating data for the model.
train_size = int(len(scaled_data)*.7)+1
val_size = int(len(scaled_data)*.2)
test_size = int(len(scaled_data)*.1)
train = scaled_data.take(train_size)
val = scaled_data.skip(train_size).take(val_size)
test = scaled_data.skip(train_size+val_size).take(test_size)

#The neural network.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
history = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])

model.save(os.path.join('models', 'elderscrollsmodel.h5'))
