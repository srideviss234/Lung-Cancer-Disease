import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, concatenate, UpSampling3D


# TransDenseBlock
def trans_dense_block(x, num_filters):
    return Conv3D(num_filters, (3, 3, 3), padding='same')(x)


# DenseBlock
def dense_block(x, num_layers, num_filters):
    for _ in range(num_layers):
        x = Conv3D(num_filters, (3, 3, 3), padding='same', activation='relu')(x)
    return x


# 3D Trans-DenseUNet++
def build_3d_trans_dense_unet(input_shape):
    # Encoder
    input_layer = Input(shape=input_shape)
    enc1 = dense_block(input_layer, 2, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(enc1)
    enc2 = dense_block(pool1, 2, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(enc2)

    # Bridge
    bridge = dense_block(pool2, 2, 128)

    # Decoder
    up1 = UpSampling3D(size=(2, 2, 2))(bridge)
    concat1 = concatenate([up1, enc2], axis=-1)
    dec1 = dense_block(concat1, 2, 64)

    up2 = UpSampling3D(size=(2, 2, 2))(dec1)
    concat2 = concatenate([up2, enc1], axis=-1)
    dec2 = dense_block(concat2, 2, 32)

    # Output
    output_layer = Conv3D(1, (1, 1, 1), activation='sigmoid')(dec2)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def Model_3D_TDUnet(image_data):
    # Assuming you have image_data for input
    # Define model and compile
    input_shape = (64, 64, 64, 3)  # Adjust dimensions as per your input data
    model = build_3d_trans_dense_unet(input_shape)
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    model.fit(image_data, image_data, epochs=5, batch_size=2)

    # Predict using the trained model
    predicted_images = model.predict(image_data)
    return predicted_images