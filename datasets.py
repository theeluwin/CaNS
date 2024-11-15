import pickle
import random

import numpy as np

from typing import Optional
from pathlib import Path

from torch import (
    Tensor,
    LongTensor,
    topk as torch_topk,
    multinomial as torch_multinomial,
)
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


__all__ = (
    'PlainTrainDataset',
    'LWPContrastiveTrainDataset',
    'EvalDataset',
)


class PlainTrainDataset(Dataset):

    data_root = Path('data')

    def __init__(self, name: str):

        # params
        self.name = name

        # load data
        with open(self.data_root / name / 'iid2iindex.pkl', 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
        with open(self.data_root / name / 'uindex2urows_train.pkl', 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)

        # settle down
        self.uindices = list(self.uindex2urows_train.keys())
        self.num_items = len(self.iid2iindex)
        self.stamp_min = 9999999999
        self.stamp_max = 0
        self.iscs = []  # type: ignore
        for _, urows in self.uindex2urows_train.items():
            for iindex, stamp, icontext in urows:
                if stamp > self.stamp_max:
                    self.stamp_max = stamp
                if stamp < self.stamp_min:
                    self.stamp_min = stamp
                self.iscs.append((iindex, stamp, icontext))
        self.stamp_interval = self.stamp_max - self.stamp_min

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):
        uindex = self.uindices[index]
        urows = self.uindex2urows_train[uindex]
        return {
            'uindex': uindex,
            'urows': urows,
        }

    @staticmethod
    def collate_fn(samples):
        uindex = [sample['uindex'] for sample in samples]
        urows = [sample['urows'] for sample in samples]
        return {
            'uindex': uindex,
            'urows': urows,
        }


class LWPContrastiveTrainDataset(Dataset):

    data_root = Path('data')

    def __init__(self,
                 name: str,
                 sequence_len: int,
                 random_cut_prob: float = 1.0,
                 replace_user_prob: float = 0.0,
                 replace_item_prob: float = 0.01,
                 train_num_negatives: int = 100,
                 clustering_method: str = 'iterative',  # 'iterative', 'kmeans', 'interval'
                 ncs: str = 'none',  # 'none', 'random', 'hard',
                 ncs_ccs: int = 1,
                 nis: str = 'uniform',  # 'negative', 'uniform', 'popular'
                 nis_ccs: int = 1,
                 random_seed: Optional[int] = None,
                 ):

        # params
        self.name = name
        self.sequence_len = sequence_len
        self.random_cut_prob = random_cut_prob
        self.replace_user_prob = replace_user_prob
        self.replace_item_prob = replace_item_prob
        self.train_num_negatives = train_num_negatives
        self.clustering_method = clustering_method
        self.ncs = ncs
        self.ncs_ccs = ncs_ccs
        self.nis = nis
        self.nis_ccs = nis_ccs
        self.random_seed = random_seed

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(self.data_root / name / 'uid2uindex.pkl', 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(self.data_root / name / 'iid2iindex.pkl', 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
        with open(self.data_root / name / 'uindex2urows_train.pkl', 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)

        # settle down
        self.uindices = []
        iindexset_train = set()
        self.uindex2iindexset = {}
        self.stamp_min = 9999999999
        self.stamp_max = 0
        self.iscs = []  # type: ignore
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()
            for iindex, stamp, icontext in urows:
                iindexset_user.add(iindex)
                iindexset_train.add(iindex)
                if stamp > self.stamp_max:
                    self.stamp_max = stamp
                if stamp < self.stamp_min:
                    self.stamp_min = stamp
                self.iscs.append((iindex, stamp, icontext))
            self.uindex2iindexset[uindex] = iindexset_user
            if len(urows) < 2:
                continue
            self.uindices.append(uindex)
        self.iindices_train = list(iindexset_train)
        self.max_iindex_train = max(iindexset_train)
        self.num_items = len(self.iid2iindex)
        self.stamp_interval = self.stamp_max - self.stamp_min

        # clustering (ncs)
        if self.ncs == 'hard':
            ztamps = [self.normalize_stamp(stamp) for _, stamp, _ in self.iscs]
            self.ncs_centroids = self.temporal_clustering(ztamps, self.ncs_ccs)
            self.ncs_cluster2popularity = np.zeros((self.ncs_ccs, self.max_iindex_train + 1))
            self.ncs_cluster2icontexts = {cluster: [] for cluster in range(self.ncs_ccs)}
            for uindex, urows in self.uindex2urows_train.items():
                for iindex, stamp, icontext in urows:
                    ztamp = self.normalize_stamp(stamp)
                    cluster = self.ncs_get_cluster_from_ztamp(ztamp)
                    self.ncs_cluster2icontexts[cluster].append(icontext)
                    self.ncs_cluster2popularity[cluster, iindex] += 1

        # clustering (nis)
        if self.nis == 'popular':
            ztamps = [self.normalize_stamp(stamp) for _, stamp, _ in self.iscs]
            self.nis_centroids = self.temporal_clustering(ztamps, self.nis_ccs)
            self.nis_cluster2popularity = np.zeros((self.nis_ccs, self.max_iindex_train + 1))
            self.nis_cluster2iweights = np.zeros((self.nis_ccs, self.max_iindex_train + 1))
            for uindex, urows in self.uindex2urows_train.items():
                for iindex, stamp, icontext in urows:
                    ztamp = self.normalize_stamp(stamp)
                    cluster = self.nis_get_cluster_from_ztamp(ztamp)
                    self.nis_cluster2popularity[cluster, iindex] += 1
            for cluster in range(self.nis_ccs):
                iweights = self.nis_cluster2popularity[cluster]
                iweights = np.log(iweights + 1) + 1
                iweights = iweights / iweights.sum()
                self.nis_cluster2iweights[cluster] = iweights

        # tokens
        self.padding_token = 0

        # icontext info
        _, _, sample_icontext = self.iscs[0]
        self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def normalize_stamp(self, stamp):
        base = self.stamp_min
        interval = self.stamp_interval
        return [(stamp - base) / interval]

    def temporal_clustering(self, ztamps, ccs):

        if self.clustering_method == 'kmeans':
            X = np.array(ztamps)
            kmeans = KMeans(n_clusters=ccs).fit(X)
            predictions = kmeans.predict(X)
            centroids = kmeans.cluster_centers_

        elif self.clustering_method == 'interval':
            X = np.array(ztamps)
            X_min = X.min()
            X_max = X.max()
            centroids = np.linspace(X_min, X_max, ccs + 1).reshape(-1, 1)
            centroids = (centroids[:-1] + centroids[1:]) / 2

        else:  # iterative
            Xs = [np.array(ztamps)]
            while len(Xs) < ccs:
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
            centroids = np.array([np.array(X).mean(axis=0) for X in Xs])

        return centroids

    def ncs_get_cluster_from_ztamp(self, ztamp):
        x = np.array(ztamp)[np.newaxis, :]
        return np.power(self.ncs_centroids - x, 2).sum(axis=1).argmin()

    def nis_get_cluster_from_ztamp(self, ztamp):
        x = np.array(ztamp)[np.newaxis, :]
        return np.power(self.nis_centroids - x, 2).sum(axis=1).argmin()

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # data point
        uindex = self.uindices[index]
        urows = self.uindex2urows_train[uindex]
        iindexset_point = set(self.uindex2iindexset[uindex])
        num_iindices_train = len(self.iindices_train)

        # data driven regularization: replace user (see SSE-PT)
        if self.rng.random() < self.replace_user_prob:
            sampled_index = self.rng.randrange(0, len(self.uindices))
            uindex = self.uindices[sampled_index]

        # long sequence random cut (see SSE-PT++)
        if self.rng.random() < self.random_cut_prob:
            urows = urows[:self.rng.randint(2, len(urows))]

        # last as positive
        positive_token, positive_stamp, positive_icontext = urows[-1]
        extract_tokens = [positive_token]

        # bake profile
        profile_tokens = []
        profile_stamps = []
        profile_icontexts = []
        for profile_iindex, profile_stamp, profile_icontext in urows[:-1][-self.sequence_len:]:

            # data driven regularization: replace item (see SSE)
            if self.rng.random() < self.replace_item_prob:
                sampled_index = self.rng.randrange(0, num_iindices_train)
                profile_iindex = self.iindices_train[sampled_index]
                iindexset_point.add(profile_iindex)

            # add item
            profile_tokens.append(profile_iindex)
            profile_stamps.append(profile_stamp)
            profile_icontexts.append(profile_icontext)

        # add paddings
        _, padding_stamp, _ = urows[0]
        padding_len = self.sequence_len - len(profile_tokens)
        profile_tokens = [self.padding_token] * padding_len + profile_tokens
        profile_stamps = [padding_stamp] * padding_len + profile_stamps
        profile_icontexts = [self.padding_icontext] * padding_len + profile_icontexts

        # sample negatives
        negative_tokens = set()
        if self.nis == 'negative':
            while len(negative_tokens) < self.train_num_negatives:
                while True:
                    sampled_index = self.rng.randrange(0, num_iindices_train)
                    negative_iindex = self.iindices_train[sampled_index]
                    if negative_iindex not in iindexset_point and negative_iindex not in negative_tokens:
                        break
                negative_tokens.add(negative_iindex)
            negative_tokens = list(negative_tokens)
        else:
            if self.nis == 'uniform':
                iweights = np.ones(self.max_iindex_train + 1)
                iweights[0] = 0

                # excluding positive trick (not theorectically sound)
                iweights = iweights.copy()
                for iindex in iindexset_point:
                    iweights[iindex] = 0
                iweights = iweights / iweights.sum()

                # negative_tokens = torch_multinomial(Tensor(iweights), self.train_num_negatives, replacement=False).tolist()  # disallow duplications
                negative_tokens = torch_multinomial(Tensor(iweights), self.train_num_negatives, replacement=True).tolist()  # allow duplications

            elif self.nis == 'popular':
                ztamp = self.normalize_stamp(positive_stamp)
                cluster = self.nis_get_cluster_from_ztamp(ztamp)
                iweights = self.nis_cluster2iweights[cluster]
                noise = np.random.random(self.max_iindex_train + 1)
                iscores = np.power(noise, 1 / iweights)
                iscores[0] = 0
                for iindex in iindexset_point:
                    iscores[iindex] = 0
                negative_tokens = torch_topk(Tensor(iscores), self.train_num_negatives).indices.tolist()

        extract_tokens.extend(negative_tokens)

        # fill extract
        extract_stamps = []
        extract_icontexts = []
        if self.ncs == 'none':
            for _ in extract_tokens:
                extract_stamps.append(positive_stamp)
                extract_icontexts.append(positive_icontext)
        else:
            num_normal_negatives = self.train_num_negatives - 1
            extract_tokens = extract_tokens[:num_normal_negatives]
            for _ in range(num_normal_negatives):
                extract_stamps.append(positive_stamp)
                extract_icontexts.append(positive_icontext)
            extract_tokens.append(positive_token)
            if self.ncs == 'random':
                while True:
                    sampled_index = self.rng.randrange(0, len(self.iscs))
                    negative_iindex, negative_stamp, negative_icontext = self.iscs[sampled_index]
                    if negative_iindex != positive_token:
                        break
                extract_stamps.append(negative_stamp)
                extract_icontexts.append(negative_icontext)
            elif self.ncs == 'hard':
                cluster_dist = self.ncs_cluster2popularity[:, positive_token]
                cweights = cluster_dist + 1
                cweights = np.log(cweights) + 1
                cweights = cweights / cweights.sum()
                while True:
                    noise = np.random.random(self.ncs_ccs)
                    cscores = np.power(noise, cweights)
                    cluster = np.argmax(cscores)
                    candidates = self.ncs_cluster2icontexts[cluster]  # type: ignore
                    if len(candidates):
                        break
                sampled_index = self.rng.randrange(0, len(candidates))
                negative_icontext = candidates[sampled_index]
                extract_stamps.append(positive_stamp)  # just a filler
                extract_icontexts.append(negative_icontext)

        # normal
        if True:
            t_profile_icontexts = Tensor(np.array(profile_icontexts))
            t_extract_icontexts = Tensor(np.array(extract_icontexts))

        # motiv: no context
        if False:
            t_profile_icontexts = Tensor(self.padding_icontext).view(1, -1).repeat(len(profile_icontexts), 1)
            t_extract_icontexts = Tensor(self.padding_icontext).view(1, -1).repeat(len(extract_icontexts), 1)

        # motiv: random context
        if False:
            t_profile_icontexts = Tensor(np.array(profile_icontexts))
            while True:
                sampled_index = self.rng.randrange(0, len(self.iscs))
                negative_iindex, _, negative_icontext = self.iscs[sampled_index]
                if negative_iindex != positive_token:
                    break
            t_extract_icontexts = Tensor(negative_icontext).view(1, -1).repeat(len(extract_icontexts), 1)

        # motiv: wrong context
        if False:
            t_profile_icontexts = Tensor(np.array(profile_icontexts))
            extract_icontexts = []
            for _ in extract_tokens:
                while True:
                    sampled_index = self.rng.randrange(0, len(self.iscs))
                    negative_iindex, _, negative_icontext = self.iscs[sampled_index]
                    if negative_iindex != positive_token:
                        break
                extract_icontexts.append(negative_icontext)
            t_extract_icontexts = Tensor(np.array(extract_icontexts))

        # return tensorized data point
        return {
            'uindex': uindex,
            'profile_tokens': LongTensor(profile_tokens),
            'profile_stamps': LongTensor(profile_stamps),
            'profile_icontexts': t_profile_icontexts,
            'extract_tokens': LongTensor(extract_tokens),
            'extract_stamps': LongTensor(extract_stamps),
            'extract_icontexts': t_extract_icontexts,
            'label': 0,
        }


class EvalDataset(Dataset):

    data_root = Path('data')

    def __init__(self,
                 name: str,
                 target: str,  # 'valid', 'test'
                 ns: str,  # 'random', 'popular'
                 sequence_len: int,
                 valid_num_negatives: int = 100,
                 random_seed: Optional[int] = None,
                 ):

        # params
        self.name = name
        self.target = target
        self.ns = ns
        self.sequence_len = sequence_len
        self.valid_num_negatives = valid_num_negatives
        self.random_seed = random_seed

        # rng
        if random_seed is None:
            self.rng = random.Random()
        else:
            self.rng = random.Random(random_seed)

        # load data
        with open(self.data_root / name / 'uid2uindex.pkl', 'rb') as fp:
            self.uid2uindex = pickle.load(fp)
            self.uindex2uid = {uindex: uid for uid, uindex in self.uid2uindex.items()}
        with open(self.data_root / name / 'iid2iindex.pkl', 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.iindex2iid = {iindex: iid for iid, iindex in self.iid2iindex.items()}
        with open(self.data_root / name / 'uindex2urows_train.pkl', 'rb') as fp:
            self.uindex2urows_train = pickle.load(fp)
            self.iindexset_train = set()
            for uindex, urows in self.uindex2urows_train.items():
                for iindex, _, _ in urows:
                    self.iindexset_train.add(iindex)
        with open(self.data_root / name / 'uindex2urows_valid.pkl', 'rb') as fp:
            self.uindex2urows_valid = pickle.load(fp)
            self.iindexset_valid = set()
            for uindex, urows in self.uindex2urows_valid.items():
                for iindex, _, _ in urows:
                    self.iindexset_valid.add(iindex)
        with open(self.data_root / name / 'uindex2urows_test.pkl', 'rb') as fp:
            self.uindex2urows_test = pickle.load(fp)
            self.uindex2aiindexset_test = {}
            for uindex, urows in self.uindex2urows_test.items():
                aiindexset = set()
                for iindex, _, _ in urows:
                    aiindexset.add(iindex)
                self.uindex2aiindexset_test[uindex] = aiindexset
        with open(self.data_root / name / f'ns_{ns}.pkl', 'rb') as fp:
            self.uindex2negatives = pickle.load(fp)

        # settle down
        if target == 'valid':
            self.uindices = []
            for uindex in self.uindex2urows_valid:
                if uindex in self.uindex2urows_train:
                    self.uindices.append(uindex)
        elif target == 'test':
            self.uindices = []
            for uindex in self.uindex2aiindexset_test:
                if uindex not in self.uindex2urows_train and uindex not in self.uindex2urows_valid:
                    continue
                self.uindices.append(uindex)
        self.iindexset_known = set()
        self.uindex2iindexset = {}
        for uindex, urows in self.uindex2urows_train.items():
            iindexset_user = set()
            for iindex, _, _ in urows:
                iindexset_user.add(iindex)
                self.iindexset_known.add(iindex)
            self.uindex2iindexset[uindex] = iindexset_user
        for uindex, urows in self.uindex2urows_valid.items():
            iindexset_user = set()
            for iindex, _, _ in urows:
                iindexset_user.add(iindex)
                self.iindexset_known.add(iindex)
            if uindex not in self.uindex2iindexset:
                self.uindex2iindexset[uindex] = set()
            self.uindex2iindexset[uindex] |= iindexset_user
        self.iindices_known = list(self.iindexset_known)
        self.num_items = len(self.iid2iindex)

        # tokens
        self.padding_token = 0

        # icontext info
        _, _, sample_icontext = urows[0]
        self.icontext_dim = len(sample_icontext)
        self.padding_icontext = np.zeros(self.icontext_dim)

    def __len__(self):
        return len(self.uindices)

    def __getitem__(self, index):

        # get data point
        uindex = self.uindices[index]
        urows_train = self.uindex2urows_train.get(uindex, [])
        urows_valid = self.uindex2urows_valid.get(uindex, [])
        urows_test = self.uindex2urows_test.get(uindex, [])

        # prepare rows
        if self.target == 'valid':
            urows_known = urows_train
            urows_eval = urows_valid
        elif self.target == 'test':
            urows_known = urows_train + urows_valid
            urows_eval = urows_test

        # get eval row
        answer_iindex, answer_stamp, answer_icontext = urows_eval[0]
        extract_tokens = [answer_iindex]

        # bake profile
        profile_tokens = []
        profile_stamps = []
        profile_icontexts = []
        for profile_iindex, profile_stamp, profile_icontext in urows_known[-self.sequence_len:]:
            profile_tokens.append(profile_iindex)
            profile_stamps.append(profile_stamp)
            profile_icontexts.append(profile_icontext)

        # add paddings
        _, padding_stamp, _ = urows_known[0]
        padding_len = self.sequence_len - len(profile_tokens)
        profile_tokens = [self.padding_token] * padding_len + profile_tokens
        profile_stamps = [padding_stamp] * padding_len + profile_stamps
        profile_icontexts = [self.padding_icontext] * padding_len + profile_icontexts

        # sample negatives
        if self.target == 'valid':
            negative_tokens = set()
            iindexset_user = self.uindex2iindexset[uindex]
            num_iindices_known = len(self.iindices_known)
            while len(negative_tokens) < self.valid_num_negatives:
                while True:
                    sampled_index = self.rng.randrange(0, num_iindices_known)
                    negative_iindex = self.iindices_known[sampled_index]
                    if negative_iindex not in iindexset_user and negative_iindex not in negative_tokens:
                        break
                negative_tokens.add(negative_iindex)
            negative_tokens = list(negative_tokens)
        elif self.target == 'test':
            negative_tokens = self.uindex2negatives[uindex]
        extract_tokens.extend(negative_tokens)

        # bake extract
        extract_stamps = []
        extract_icontexts = []
        for _ in extract_tokens:
            extract_stamps.append(answer_stamp)
            extract_icontexts.append(answer_icontext)
        labels = [1] + [0] * (len(extract_tokens) - 1)

        # return tensorized data point
        return {
            'uindex': uindex,
            'profile_tokens': LongTensor(profile_tokens),
            'profile_stamps': Tensor(np.array(profile_stamps)),
            'profile_icontexts': Tensor(np.array(profile_icontexts)),
            'extract_tokens': LongTensor(extract_tokens),
            'extract_stamps': Tensor(np.array(extract_stamps)),
            'extract_icontexts': Tensor(np.array(extract_icontexts)),
            'labels': LongTensor(labels),
        }
