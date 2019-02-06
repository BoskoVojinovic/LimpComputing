import os

from keras import Sequential
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.layers import MaxPooling2D, Activation, BatchNormalization, Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
import cv2 as cv


class CNN:
    def __init__(self, file):
        if os.path.isfile(file):
            model = load_model(file)
            self._model = model
            print('CNN: model imported!')
            return

        print('CNN: model missing, generating!')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))

        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                 height_shift_range=0.08, zoom_range=0.08)
        test_gen = ImageDataGenerator()
        train_generator = gen.flow(x_train, y_train, batch_size=64)
        test_generator = test_gen.flow(x_test, y_test, batch_size=64)

        model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5,
                            validation_data=test_generator, validation_steps=10000//64)

        print('===========================')
        score = model.evaluate(x_test, y_test)
        print('CNN: Accuracy: ', score[1] * 100, '%')
        print('===========================')

        model.save(file)
        self._model = model

    def prepare(self, region):
        region = cv.resize(region, (28,28))
        region = region.astype('float32')
        region /= 255
        region = region.reshape(1, 28, 28, 1)
        return region

    def predict(self, region):
        return self._model.predict_classes(region)
