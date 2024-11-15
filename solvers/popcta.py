import numpy as np

from typing import (
    List,
    Dict,
)
from shutil import copyfile

from torch import Tensor
from sklearn.cluster import KMeans  # type: ignore

from .pop import PopSolver


__all__ = (
    'PopCTASolver',
)


# clustering-based time-aware popularity
class PopCTASolver(PopSolver):

    # override
    def init_model(self) -> None:

        # cluster data
        ztamps = []
        for _, stamp, _ in self.train_dataset.iscs:  # type: ignore
            ztamps.append(self.normalize_stamp(stamp))  # type: ignore

        # clustering
        self.num_ccs = self.config['dataloader']['num_ccs']
        Xs = [np.array(ztamps)]
        while len(Xs) < self.num_ccs:
            target = np.argmax([len(X) for X in Xs])
            X = Xs.pop(target)
            X = X.flatten()
            X.sort()
            half = len(X) // 2
            X1 = X[:half]
            X2 = X[half:]
            init = np.array([X1.mean(), X2.mean()]).reshape(-1, 1)
            X = X[:, np.newaxis]
            kmeans = KMeans(n_clusters=2, init=init).fit(X)
            predictions = kmeans.predict(X)
            X1 = X[predictions == 0]
            X2 = X[predictions == 1]
            Xs.append(X1)
            Xs.append(X2)
        self.centroids = np.array([np.array(X).mean(axis=0) for X in Xs])

        # bowl
        self.model: Dict[int, List[int]] = {}  # type: ignore
        for clusters in range(self.num_ccs):
            iindex2popularity = [0 for _ in range(self.num_items + 1)]
            self.model[clusters] = iindex2popularity

    def normalize_stamp(self, stamp):
        base = self.train_dataset.stamp_min  # type: ignore
        interval = self.train_dataset.stamp_interval  # type: ignore
        return [(stamp - base) / interval]

    def get_cluster_from_ztamp(self, ztamp):
        x = np.array(ztamp)[np.newaxis, :]
        return np.power(self.centroids - x, 2).sum(axis=1).argmin()

    # override (special: inplace update)
    def calculate_loss(self, batch):

        for _, urows in zip(batch['uindex'], batch['urows']):

            # just plain counting
            for iindex, stamp, _ in urows:
                ztamp = self.normalize_stamp(stamp)
                cluster = self.get_cluster_from_ztamp(ztamp)
                self.model[cluster][iindex] += 1

        return 0.0

    # override
    def calculate_rankers(self, batch):

        # forward
        scores = []
        for extract_stamps in batch['extract_stamps'].numpy():
            ztamp = self.normalize_stamp(extract_stamps[0])
            cluster = self.get_cluster_from_ztamp(ztamp)
            scores.append(self.model[cluster])
        scores = Tensor(np.array(scores))

        # get rankers
        candidates = batch['extract_tokens'].to(self.device)  # b x C
        rankers = scores.gather(1, candidates).argsort(dim=1, descending=True)

        return rankers

    # override
    def backup(self):
        copyfile('solvers/popcta.py', self.data_dir / 'solver.py')
