import csv
import os
from time import time

import keras
import keras.backend as K
import numpy as np
from keras.engine.topology import InputSpec, Layer
from keras.losses import mean_squared_error
from keras.models import Model
from sklearn.cluster import KMeans

from .model import CAE


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            name="clusters",
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (
            1.0
            + (
                K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)
                / self.alpha
            )
        )
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {"n_clusters": self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCEC:
    def __init__(
        self,
        input_shape,
        filters,
        n_clusters,
        lambda_kl=0.05,
        alpha=1.0,
        batch_size=1000,
        epochs=100,
        maxiter=2e4,
        update_interval=140,
        cae_weights=None,
        save_dir="dcec",
    ):
        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.lambda_kl = lambda_kl
        self.alpha = alpha
        self.pretrained = False
        self.batch_size = batch_size
        self.epochs = epochs
        self.maxiter = maxiter
        self.update_interval = update_interval
        self.cae_weights = cae_weights
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name="embedding").output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(hidden)
        self.model = Model(
            inputs=self.cae.input, outputs=[clustering_layer, self.cae.output]
        )

    def pretrain(self, x, batch_size, epochs, optimizer="adam", save_dir="dcec"):
        print("...Pretraining...")
        self.cae.compile(optimizer=optimizer, loss="mse")
        from keras.callbacks import CSVLogger

        csv_logger = CSVLogger(save_dir + "/pretrain_log.csv")

        # begin training
        t0 = time()
        self.cae.fit(
            x,
            x,
            verbose=0,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[csv_logger],
        )
        print("Pretraining time: ", time() - t0)
        self.cae.save(save_dir + "/pretrain_cae_model.h5")
        print("Pretrained weights are saved to %s/pretrain_cae_model.h5" % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        self.X = x.reshape(-1, *self.input_shape)
        return self.encoder.predict(self.X)

    def predict(self, x):
        return self.get_q(x).argmax(1)

    def get_q(self, x):
        self.X = x.reshape(-1, *self.input_shape)
        q, _ = self.model.predict(self.X, verbose=0)
        return q

    def score_samples(self, x):
        self.X = x.reshape(-1, *self.input_shape)
        y = self.cae.predict(self.X)
        re = keras.losses.mean_squared_error(
            x.reshape(-1, np.prod(self.input_shape)),
            y.reshape(-1, np.prod(self.input_shape)),
        ).numpy()
        scores = np.amax(self.get_q(x), 1)
        return re, scores

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=["kld", "mse"], loss_weights=[1, 1], optimizer="adam"):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x):
        x = x.reshape(-1, *self.input_shape)
        self.X = x
        print("Update interval", self.update_interval)
        save_interval = x.shape[0] / self.batch_size * 5
        print("Save interval", save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and self.cae_weights is None:
            print("...pretraining CAE using default hyper-parameters:")
            print("   optimizer='adam';   epochs=200")
            self.pretrain(x, self.batch_size, self.epochs, save_dir=self.save_dir)
            self.pretrained = True
        elif self.cae_weights is not None:
            self.cae.load_weights(self.cae_weights)
            print("cae_weights is loaded successfully.")

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print("Initializing cluster centers with k-means.")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(self.encoder.predict(x))
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        self.compile(loss_weights=[self.lambda_kl, 1])

        t2 = time()
        self.loss_evolution = []
        index = 0
        for ite in range(int(self.maxiter)):
            if ite % self.update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(
                    q
                )  # update the auxiliary target distribution p
                self.y_pred = q.argmax(1)

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                self.loss_evolution.append(
                    self.model.train_on_batch(
                        x=x[index * self.batch_size : :],
                        y=[
                            p[index * self.batch_size : :],
                            x[index * self.batch_size : :],
                        ],
                    )
                )
                index = 0
            else:
                self.loss_evolution.append(
                    self.model.train_on_batch(
                        x=x[index * self.batch_size : (index + 1) * self.batch_size],
                        y=[
                            p[index * self.batch_size : (index + 1) * self.batch_size],
                            x[index * self.batch_size : (index + 1) * self.batch_size],
                        ],
                    )
                )
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                print(
                    "saving model to:",
                    self.save_dir + "/dcec_model_" + str(ite) + ".h5",
                )
                self.model.save_weights(
                    self.save_dir + "/dcec_model_" + str(ite) + ".h5"
                )

            ite += 1

        # save the trained model
        print("saving model to:", self.save_dir + "/dcec_model_final.h5")
        self.model.save_weights(self.save_dir + "/dcec_model_final.h5")
        t3 = time()
        print("Pretrain time:  ", t1 - t0)
        print("Clustering time:", t3 - t1)
        print("Total time:     ", t3 - t0)

        _labels = self.y_pred
