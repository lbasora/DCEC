from functools import reduce

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, LocallyConnected1D
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




def buildcnn(model,filters,ns,strides,padding,input_shape=None,is1d=False,transpose=False,builder = Conv2D):
    cv = Conv2DTranspose if transpose else builder
    for i,(fi,ni,stride) in enumerate(zip(filters,ns,strides)):
        args ={}
        args["filters"] = fi
        args["kernel_size"] = (1,ni) if is1d else ni
        args["strides"]=(1,stride)  if is1d else stride
        args["padding"]=padding
        args["activation"]="relu"
        if i==0 and input_shape is not None:
            args["input_shape"]=input_shape
        if transpose:
            args["output_padding"]=(0,0)
        model.add(cv(**args))

def CAE1d(input_shape, filters, kernels):
    model = Sequential()

#    n1 = 32
#    n2 = 16
#    stride = 1
    padding = "valid"
#    filters = [8, 16,32]
#    ns = list(reversed(filters))
#    ns = [32,16,8]
    strides = [1]*len(kernels)
    buildcnn(model,filters,kernels,strides,padding,input_shape,is1d=True)
    # model.add(
    #     Conv2D(
    #         filters[0],
    #         (1,n1),
    #         strides=(1,stride),
    #         padding=padding,
    #         activation="relu",
    #         input_shape=input_shape,
    #     )
    # )
    # model.add(
    #     Conv2D(
    #         filters[1], (1,n2), strides=(1,stride), padding=padding, activation="relu", name="conv3"
    #     )
    # )
    c2s = model.layers[-1].output_shape[1:]#model.get_layer("conv3").output_shape[1:]
    model.add(Flatten())
    model.add(Dense(units=2, name="embedding"))

    # decoder

    print("c2s",c2s)#,model.layers[-1].output_shape[1:])
    model.add(Dense(units=reduce((lambda x, y: x * y), c2s), activation="relu",))

    model.add(Reshape(((c2s))))
    # model.add(
    #     Conv2DTranspose(
    #         filters[1], 3, strides=2, padding="same", activation="relu", name="deconv3"
    #     )
    # )
    buildcnn(model,reversed(filters[:-1]),reversed(kernels[1:]),reversed(strides[1:]),padding,input_shape=None,is1d=True,transpose=True)
    model.add(
        Conv2DTranspose(input_shape[-1], (1,kernels[0]), strides=(1,strides[0]), output_padding=(0,0), padding=padding)
    )
    # model.add(
    #     Conv2DTranspose(
    #         filters[0], (1,n2), strides=(1,stride), padding=padding,output_padding=(0,0), activation="relu"
    #     )
    # )
    # model.add(
    #     Conv2DTranspose(input_shape[2], (1,n1), strides=(1,stride),output_padding=(0,0), padding=padding)
    # )


    #    print("input_shape[2]",input_shape[2])
    model.summary()
    return model



def local1d(input_shape, filters, kernels):
    model = Sequential()

    padding = "valid"
    strides = [1]*len(kernels)
    for i,(fi,ni,stride) in enumerate(zip(filters,kernels,strides)):
        args ={}
        args["filters"] = fi
        args["kernel_size"] = ni
        args["strides"]=stride
        args["padding"]=padding
        args["activation"]="relu"
        if i==0 and input_shape is not None:
            args["input_shape"]=input_shape
        model.add(LocallyConnected1D(**args))

    c2s = model.layers[-1].output_shape[1:]
    model.add(Flatten())
    model.add(Dense(units=2, name="embedding"))

    # decoder

    print("c2s",c2s)#,model.layers[-1].output_shape[1:])
    model.add(Dense(units=reduce((lambda x, y: x * y), c2s), activation="relu",))

    model.add(Reshape((1,)+(c2s)))
    buildcnn(model,reversed(filters[:-1]),reversed(kernels[1:]),reversed(strides[1:]),padding,input_shape=None,is1d=True,transpose=True)
    model.add(
        Conv2DTranspose(input_shape[-1], (1,kernels[0]), strides=(1,strides[0]), output_padding=(0,0), padding=padding)
    )
    model.add(Reshape(input_shape))
    model.summary()
    return model



def dense(input_shape,archidense):
    model = Sequential()
    n = len(archidense)
    batchnorm = False
    model.add(
        Dense(
            archidense[0],
            activation="relu",
            input_shape=input_shape,
        )
    )
    if batchnorm:
        model.add(BatchNormalization())
    for i in range(1,n):
        model.add(
            Dense(
                archidense[i],
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
    for i in range(n-1,-1,-1):
        model.add(
            Dense(
                archidense[i],
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

