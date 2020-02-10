from functools import reduce

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization
from keras.models import Model, Sequential


def CAE(input_shape, filters):
    model = Sequential()

    model.add(
        Conv2D(
            filters[0],
            5,
            strides=2,
            padding="same",
            activation="relu",
            name="conv1",
            input_shape=input_shape,
        )
    )

    # model.add(
    #     Conv2D(
    #         filters[1], 5, strides=2, padding="same", activation="relu", name="conv2"
    #     )
    # )

    model.add(
        Conv2D(
            filters[1], 3, strides=2, padding="same", activation="relu", name="conv3"
        )
    )

    model.add(Flatten())
    model.add(Dense(units=filters[2], name="embedding"))

    # decoder
    c2s = model.get_layer("conv3").output_shape[1:]
    model.add(Dense(units=reduce((lambda x, y: x * y), c2s), activation="relu",))

    model.add(Reshape(((c2s))))
    # model.add(
    #     Conv2DTranspose(
    #         filters[1], 3, strides=2, padding="same", activation="relu", name="deconv3"
    #     )
    # )

    model.add(
        Conv2DTranspose(
            filters[0], 5, strides=2, padding="same", activation="relu", name="deconv2"
        )
    )

    model.add(
        Conv2DTranspose(input_shape[2], 5, strides=2, padding="same", name="deconv1")
    )
    model.summary()
    return model



def CAE1d(input_shape, filters):
    model = Sequential()

    n1 = 16
    n2 = 8
    stride = 4
    padding = "valid"
    model.add(
        Conv2D(
            filters[0],
            (1,n1),
            strides=(1,stride),
            padding=padding,
            activation="relu",
            input_shape=input_shape,
        )
    )

    # model.add(
    #     Conv2D(
    #         filters[1], 5, strides=2, padding="same", activation="relu", name="conv2"
    #     )
    # )

    model.add(
        Conv2D(
            filters[1], (1,n2), strides=(1,stride), padding=padding, activation="relu"
        )
    )

    model.add(Flatten())
    model.add(Dense(units=filters[2], name="embedding"))

    # decoder
    c2s = model.get_layer("conv3").output_shape[1:]
    print("c2s",c2s)
    model.add(Dense(units=reduce((lambda x, y: x * y), c2s), activation="relu",))

    model.add(Reshape(((c2s))))
    # model.add(
    #     Conv2DTranspose(
    #         filters[1], 3, strides=2, padding="same", activation="relu", name="deconv3"
    #     )
    # )

    model.add(
        Conv2DTranspose(
            filters[0], (1,n2), strides=(1,stride), padding=padding,output_padding=(0,0), activation="relu"
        )
    )
    print("input_shape[2]",input_shape[2])
    model.add(
        Conv2DTranspose(input_shape[2], (1,n1), strides=(1,stride),output_padding=(0,0), padding=padding)
    )
    model.summary()
    return model



def dense(input_shape,filters):
    model = Sequential()
    n = 3
    batchnorm = False
    model.add(
        Dense(
            input_shape[0]//2,
            activation="relu",
            input_shape=input_shape,
        )
    )
    if batchnorm:
        model.add(BatchNormalization())
    for i in range(2,n+1):
        model.add(
            Dense(
                input_shape[0]//2**i,
                activation="relu",
            )
        )
        if batchnorm:
            model.add(BatchNormalization())
    model.add(
        Dense(
            2,
            activation=None,
            name="embedding",
        )
    )
    for i in range(n,0,-1):
        model.add(
            Dense(
                input_shape[0]//2**i,
                activation="relu",
            )
        )
        if batchnorm:
            model.add(BatchNormalization())
    model.add(
        Dense(
            input_shape[0],
            activation=None,
        )
    )
    # model.add(
    #     Conv2D(
    #         filters[1], 5, strides=2, padding="same", activation="relu", name="conv2"
    #     )
    # )
    model.summary()
    return model

