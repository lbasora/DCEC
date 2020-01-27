import numpy as np
from pathlib import Path
from keras.models import Model
from keras.utils.vis_utils import plot_model

from dcec.model import CAE

if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=Path, default="data/lszh_dcec.npy")
    parser.add_argument("--n_clusters", default=4, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--save_dir", default="dcec", type=str)
    args = parser.parse_args()
    print(args)

    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    x = np.load(args.data_path)

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 2])
    plot_model(
        model, to_file=args.save_dir + "/pretrain-model.png", show_shapes=True,
    )

    # compile the model and callbacks
    optimizer = "adam"
    model.compile(optimizer=optimizer, loss="mse")
    from keras.callbacks import CSVLogger

    csv_logger = CSVLogger(args.save_dir + "/pretrain-log.csv")

    # begin training
    t0 = time()
    model.fit(
        x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger]
    )
    print("Training time: ", time() - t0)
    model.save(args.save_dir + "/pretrain-model.h5")

    # extract features
    feature_model = Model(
        inputs=model.input, outputs=model.get_layer(name="embedding").output
    )
    features = feature_model.predict(x)
    print("feature shape=", features.shape)
