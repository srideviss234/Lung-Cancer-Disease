from Evaluation import evaluation
from keras import layers, models, metrics
import numpy as np


def conv3d_bn(x, filters, kernel_size, padding='same', strides=(1, 1, 1)):
    x = layers.Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def MDDNet_ASPP(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    # Feature extraction (Densenet-like)
    base_model = models.Sequential()
    x = conv3d_bn(input_layer, 64, (3, 3, 3), padding='same')
    x = layers.MaxPooling3D((2, 2, 2))(x)

    for _ in range(4):  # Example: 4 dense blocks
        for _ in range(4):  # Example: 4 layers per dense block
            residual = x
            x = conv3d_bn(x, 64, (1, 1, 1))
            x = conv3d_bn(x, 64, (3, 3, 3))
            x = layers.Concatenate()([x, residual])

    # Multiscale Dilated Convolution (2D-3D)
    x_2d = layers.Conv2D(64, (3, 3), dilation_rate=(2, 2), padding='same')(x)  # Example 2D dilation
    x_3d = conv3d_bn(x, 64, (3, 3, 3), padding='same')  # Example 3D dilation  # , dilation_rate=(2, 2, 2)
    x = layers.Concatenate()([x_2d, x_3d])

    # Atrous Spatial Pyramid Pooling (ASPP)
    pool1x1 = conv3d_bn(x, 64, (1, 1, 1))
    pool3x3_1 = conv3d_bn(x, 64, (3, 3, 3), padding='same')  # , dilation_rate=(6, 6, 6)
    pool3x3_2 = conv3d_bn(x, 64, (3, 3, 3), padding='same')  # , dilation_rate=(12, 12, 12)
    pool3x3_3 = conv3d_bn(x, 64, (3, 3, 3), padding='same')  # , dilation_rate=(18, 18, 18)
    x = layers.Concatenate()([pool1x1, pool3x3_1, pool3x3_2, pool3x3_3])

    # Classifier
    x = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)
    return model

def cross_entropy(y, y_pred):

    return - np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / np.size(y)

def mae(y, y_pred):
    return np.sum(np.abs(y - y_pred)) / np.size(y)

def my_custom_loss(y_true, y_pred):
    MAE = mae(y_true, y_pred)
    crossentropy = cross_entropy(y_true, y_pred)
    return MAE + crossentropy


def Model_MDDNet_ASPP(Train_Data, Train_Target, Test_Data, Test_Target, Batchsize=None):
    if Batchsize is None:
        Batchsize = 4

    # Example usage
    input_shape = (16, 16, 16, 3)  # Adjust dimensions as per your input data
    num_classes = Train_Target.shape[-1]  # Adjust based on your classification task
    model = MDDNet_ASPP(input_shape, num_classes)

    # Define the Mean Absolute Error loss
    mae_loss = metrics.MeanAbsoluteError()
    # Define model and compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', mae_loss])

    IMG_SIZE = [16, 16, 16, 3]
    Train_x = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], IMG_SIZE[3]))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE[0] * IMG_SIZE[1] * IMG_SIZE[2], 3))
        Train_x[i] = np.reshape(temp, (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], IMG_SIZE[3]))
    for i in range(Test_Data.shape[0]):
        temp_1 = np.resize(Test_Data[i], (IMG_SIZE[0] * IMG_SIZE[1] * IMG_SIZE[2], 3))
        Test_X[i] = np.reshape(temp_1, (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 3))

    Train_y = np.zeros((Train_Target.shape[0], 8, 8, 8, 4))
    Test_y = np.zeros((Test_Target.shape[0],  8, 8, 8, 4))

    # Train the model
    model.fit(Train_x, Train_y, validation_data=(Test_X, Test_y), epochs=5, batch_size=Batchsize)

    # Evaluate the model
    pred = model.predict(Test_X)
    Eval = evaluation(pred, Test_y)
    return Eval, pred
