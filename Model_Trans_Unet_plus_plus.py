from tensorflow import keras
from keras import layers
import numpy as np
import cv2 as cv


def TUnetPP(input_shape, num_classes):
    # Input layer
    inputs = keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Middle
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up3 = layers.UpSampling2D(size=(2, 2))(conv4)
    up3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    merge3 = layers.concatenate([conv3, up3], axis=3)

    up2 = layers.UpSampling2D(size=(2, 2))(merge3)
    up2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    merge2 = layers.concatenate([conv2, up2], axis=3)

    up1 = layers.UpSampling2D(size=(2, 2))(merge2)
    up1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    merge1 = layers.concatenate([conv1, up1], axis=3)

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(merge1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def Model_Trans_Unet_plus_plus(Data, Target):
    input_shape = (256, 256, 3)  # Adjust input shape as needed
    num_classes = 3  # Modify the number of classes
    tar = []
    for i in range(Target.shape[0]):
        targ = cv.cvtColor(Target[i], cv.COLOR_GRAY2RGB)
        tar.append(targ)
    tar = np.asarray(tar)
    model = TUnetPP(input_shape, num_classes)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model with your data
    model.fit(Data, tar, batch_size=4, epochs=10)
    pred = model.predict(Data)
    return pred

