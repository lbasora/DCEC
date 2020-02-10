import math

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic

from .clustering import DCEC


def input_shape(nb_samples, nb_features):
    height = [
        d for d in range(2, int(math.sqrt(nb_samples)) + 1) if nb_samples % d == 0
    ][-1]
    width = nb_samples // height
    return (width, height, nb_features)


def input_shape1d(nb_samples, nb_features):
    return (1, nb_samples, nb_features)

def pretrained_clust(
    traffic_file, list_features, filters, n_clusters, pretrained_path, to_pickle,
):
    t = Traffic.from_file(traffic_file)
    nb_samples = len(t[0])
    nb_features = len(list_features)

    dcec = DCEC(
        input_shape=input_shape(nb_samples, nb_features),
        filters=filters,
        n_clusters=n_clusters,
    )
    dcec.load_weights(pretrained_path)

    t_c = t.clustering(
        nb_samples=None,
        features=list_features,
        clustering=dcec,
        transform=MinMaxScaler(feature_range=(-1, 1)),
    ).predict()

    re = dcec.score_samples(dcec.X)
    re = MinMaxScaler(feature_range=(0, 1)).fit_transform(re.reshape(-1, 1)).flatten()
    t_c_re = pd.DataFrame.from_records(
        [dict(flight_id=f.flight_id, re=re) for f, re in zip(t_c, re)]
    )
    t_c_re = t_c.merge(t_c_re, on="flight_id")

    t_c_re.to_pickle(to_pickle)
    print(
        t_c_re.groupby(["cluster"]).agg(
            {"flight_id": "nunique", "re": ["mean", "min", "max"]}
        )
    )
    return dcec, t_c_re
