import argparse
import os
from pathlib import Path

import numpy as np

from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic

from dcec.utils import input_shape, input_shape1d, input_shape_dense
from dcec.clustering import DCEC
from dcec.model import CAE, CAE1d, dense

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=Path, default="data_64/lszh.parquet")
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--maxiter", default=10000, type=int)
    parser.add_argument("--update_interval", default=140, type=int)
    parser.add_argument("--save_dir", default="results/")
    parser.add_argument("--model", default="dense")
    parser.add_argument("--lambda_kl", type=float, default=0.)
    parser.add_argument("--filters", type=str, default="8_16_32")
    parser.add_argument("--ns", type=str, default="8_16_32")
    args = parser.parse_args()
    if args.model == "dense":
        sel_input_shape, sel_cae = input_shape_dense, dense
    elif args.model == "cae2d":
        sel_input_shape, sel_cae = input_shape, CAE
    elif args.model == "cae1d":
        sel_input_shape, sel_cae = input_shape1d, CAE1d
    else:
        raise Exception("bad model:", args.model)
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    t = Traffic.from_file(args.data_path)
    list_features = ["track_unwrapped", "longitude", "latitude", "altitude"]

    nb_flights = len(t)
    nb_samples = len(t[0])
    nb_features = len(list_features)
    print(f"nb_flights: {nb_flights}, nb_samples={nb_samples}")

    dcec = DCEC(
        input_shape=sel_input_shape(nb_samples, nb_features),
        filters=[8, 16],
        n_clusters=args.n_clusters,
        alpha=1.0,
        cae = sel_cae,
        lambda_kl = args.lambda_kl,
        batch_size=args.batch_size,
        epochs=args.epochs,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        cae_weights=None,
        save_dir=args.save_dir,
    )

    t_c = t.clustering(
        nb_samples=None,
        features=list_features,
        clustering=dcec,
        transform=MinMaxScaler(feature_range=(-1, 1)),
    ).fit_predict()

    q, _ = dcec.model.predict(dcec.X, verbose=0)
    p = dcec.target_distribution(q)  # update the auxiliary target distribution p
    re, scores = dcec.score_samples(dcec.X)
    loss = dcec.model.test_on_batch(
                x=dcec.X,
                y=[p,dcec.X,],
            )
    print(np.min(loss),len(loss),np.mean(re),scores)
    #print(t_c.groupby(["cluster"]).agg({"flight_id": "nunique"}))

#    t_c.to_pickle(f"{args.save_dir}/t_c_dcec.pkl")
