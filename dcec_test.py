import argparse
import os
from pathlib import Path

import numpy as np

from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from traffic.core import Traffic
from traffic import algorithms

from dcec.utils import input_shape, input_shape1d, input_shape_dense, input_shape_local1d
from dcec.clustering import DCEC
from dcec.model import CAE, CAE1d, dense, local1d

import generateparamlist

import tensorflow as tf
from keras import backend as K
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

#conf =K.tf.compat.v1.ConfigProto(device_count={'CPU': 1},
#                        intra_op_parallelism_threads=2,
#                        inter_op_parallelism_threads=2)
#K.set_session(tf.Session(config=conf))

# def get_trajs_from_index_list(t,indexes):
#     lres = [None] * len(indexes)
#     lindexsort = sorted(enumerate(indexes),key= lambda x:x[-1],reverse=True)
#     ires, it = lindexsort.pop()
#     for i,ti in enumerate(t):
#         if i == it:
#             lres[ires]=ti
#             if lindexsort == []:
#                 break
#             else:
#                 ires, it = lindexsort.pop()
#     return sum(lres)


def str2intlist(l):
    return [int(x) for x in l.split("_")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=Path, default="data_64/lszh.parquet")
    parser.add_argument("--n_clusters", default=5, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--maxiter", default=10000, type=int)
    parser.add_argument("--update_interval", default=140, type=int)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--model", default="dense")
    parser.add_argument("--lambda_kl", type=float, default=0.)
    parser.add_argument("--filters", type=str, default="8_16_32")
    parser.add_argument("--kernels", type=str, default="8_16_32")
    parser.add_argument("--archidense", type=str, default="8_16_32")
    parser.add_argument("--train_test", type=bool, default=True)
    parser.add_argument("--csvlog", default="toto.csv")
    parser.add_argument("--search", type=int, default=None)
    args = parser.parse_args()
    if args.search is not None:
        r = generateparamlist.testrandomgen()
        args.filters = r["--filters"][args.search]
        args.kernels = r["--kernels"][args.search]
        args.archidense = r["--archidense"][args.search]
        args.model = r["--model"][args.search]
        print("search",args.search,args.filters,args.kernels)

    filters= str2intlist(args.filters)
    kernels= str2intlist(args.kernels)
    archidense = str2intlist(args.archidense)
    if args.model == "dense":
        sel_input_shape, sel_cae = input_shape_dense, lambda input_shape: dense(input_shape,archidense)
    elif args.model == "cae2d":
        sel_input_shape, sel_cae = input_shape, lambda input_shape: CAE(input_shape,filters,kernels)
    elif args.model == "cae1d":
        sel_input_shape, sel_cae = input_shape1d, lambda input_shape: CAE1d(input_shape,filters,kernels)
    elif args.model == "local1d":
        sel_input_shape, sel_cae = input_shape_local1d, lambda input_shape: local1d(input_shape,filters,kernels)
    else:
        raise Exception("bad model:", args.model)
    print(args)

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tall = Traffic.from_file(args.data_path)#.sample(frac=1).reset_index(drop=True)
#    tall.data = tall.data.sample(frac=1).reset_index(drop=True)
    if args.train_test:
        toto = np.array([f for f in list(tall.flight_ids)])#()
        print("toto.shape",toto.shape)
        indexes = np.random.permutation(len(tall))
        ntrain = int(len(tall) * 0.9)
        t =tall[list(toto[indexes[:ntrain]])]#get_trajs_from_index_list(tall,[int(i) for i in indexes[:ntrain]])
        ttest = tall[list(toto[indexes[ntrain:]])]#get_trajs_from_index_list(tall,[int(i) for i in indexes[ntrain:]])
    else:
        t = tall
        ttest = tall
    print(tall,ttest)
#    print(t[0:1],len(t))
    list_features = ["track_unwrapped", "longitude", "latitude", "altitude"]

    nb_flights = len(t)
    nb_samples = len(t[0])
    nb_features = len(list_features)
    print(f"nb_flights: {nb_flights}, nb_samples={nb_samples}")
#    print(vars(args))
#    raise Exception
    csvlog = None if args.csvlog is None else (args.csvlog, vars(args))
    input_shape=sel_input_shape(nb_samples, nb_features)
    print("input_shape",input_shape)
    dcec = DCEC(
        input_shape=input_shape,
#        filters= [int(x) for x in args.filters.split("_")],
        n_clusters=args.n_clusters,
        alpha=1.0,
        cae = sel_cae,
        lambda_kl = args.lambda_kl,
        batch_size=args.batch_size,
        epochs=args.epochs,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        cae_weights=None,
        save_dir = args.save_dir,
        csvlog = csvlog,
    )
    transform = MinMaxScaler(feature_range=(-1, 1))
    clustering = t.clustering(
        nb_samples=None,
        features=list_features,
        clustering=dcec,
        transform=transform,
    )
    dcec.testfeatures = algorithms.clustering.prepare_features(ttest,nb_samples=clustering.nb_samples,features=clustering.features,projection = clustering.projection)
    dcec.transform = transform
    #.reshape(-1,*dcec.input_shape)
    t_c = clustering.fit_predict()
#    print(clustering.transform.min_,clustering.transform.scale_)

    def evalpred(clustering, traff):
        X = algorithms.clustering.prepare_features(traff,nb_samples=clustering.nb_samples,features=clustering.features,projection = clustering.projection)
        X = clustering.transform.transform(X).reshape(-1,*dcec.input_shape)
#        print(
        q, _ = dcec.model.predict(X, verbose=0)
        p = dcec.target_distribution(q)  # update the auxiliary target distribution p
        re, scores = dcec.score_samples(X)
        dcec.compile(loss_weights=[0,1])
        loss = dcec.model.test_on_batch(
                x=X,
                y=[p,X,],
            )
        print(loss,len(loss),np.mean(re),np.mean(scores))
        dcec.compile(loss_weights=[1,0])
        loss = dcec.model.test_on_batch(
                x=X,
                y=[p,X,],
            )
        print((loss),len(loss),np.mean(re),np.mean(scores))
#    evalpred(clustering,ttest)
    #print(t_c.groupby(["cluster"]).agg({"flight_id": "nunique"}))

#    t_c.to_pickle(f"{args.save_dir}/t_c_dcec.pkl")
