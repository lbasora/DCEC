import argparse
import os
from pathlib import Path

from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic

from dcec.utils import input_shape
from dcec.clustering import DCEC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=Path, default="data/lszh.parquet")
    parser.add_argument("--n_clusters", default=4, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--maxiter", default=1000, type=int)
    parser.add_argument("--update_interval", default=140, type=int)
    parser.add_argument("--save_dir", default="results/")
    args = parser.parse_args()
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
        input_shape=input_shape(nb_samples, nb_features),
        filters=[32, 64, 2],
        n_clusters=args.n_clusters,
        alpha=1.0,
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

    print(t_c.groupby(["cluster"]).agg({"flight_id": "nunique"}))

    t_c.to_pickle(f"{args.save_dir}/t_c_dcec.pkl")
