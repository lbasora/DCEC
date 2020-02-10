from functools import reduce

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
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

    model.add(
        Conv2D(
            filters[0],
            (1,5),
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
            filters[1], (1,3), strides=2, padding="same", activation="relu", name="conv3"
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
            filters[0], (1,3), strides=(1,2), padding="same",output_padding=(0,1), activation="relu", name="deconv2"
        )
    )
    print("input_shape[2]",input_shape[2])
    model.add(
        Conv2DTranspose(input_shape[2], (1,5), strides=(1,2),output_padding=(0,1), padding="same", name="deconv1")
    )
    model.summary()
    return model
