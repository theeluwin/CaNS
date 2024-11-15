import json
import pickle
import argparse

import numpy as np
import pandas as pd

from pathlib import Path
from random import Random
from datetime import datetime as dt

from tqdm import tqdm
from sklearn.decomposition import PCA  # type: ignore


# settings
DNAMES = (

    # movielens
    'ml1m',
    'ml20m',

    # amazon carca
    'fashion',
    'men',
    'game',
    'beauty',

    # domain specific
    'dressipi',  # fashion (recsys challenge 2022)
    'steam',  # game (ver 2, mcauley), no attrs

    # session-based
    'retailrocket',  # no attrs
    'diginetica',  # no attrs

)
CARCAS = (
    'fashion',
    'men',
    'game',
    'beauty',
)
NUM_NEGATIVE_SAMPLES = 100
USE_FILTER_OUT = True
MIN_ITEM_COUNT_PER_USER = 5
MIN_USER_COUNT_PER_ITEM = 5
USE_CUT_OUT = False
MAX_SEQUENCE_LENGTH = 400
USE_ICONTEXT_PCA = False
ICONTEXT_PCA_DIM = 15  # among 23
ML1M_MIN_RATING = 1.0
ML20M_MIN_RATING = 4.0
ICONTEXT_COLUMNS = [
    'marker',
    'year',
    'month',
    'month_cos',
    'month_sin',
    'day',
    'day_cos',
    'day_sin',
    'weekday',
    'weekday_cos',
    'weekday_sin',
    'yearday',
    'yearday_cos',
    'yearday_sin',
    'week',
    'week_cos',
    'week_sin',
    'hour',
    'hour_cos',
    'hour_sin',
    'minutesecond',
    'minutesecond_cos',
    'minutesecond_sin',
]


def parse_args():

    # constants
    tasks = {
        'prepare',
        'count_stats',
    }

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=tasks, help="task to do")
    parser.add_argument('--dname', type=str, choices=DNAMES, help="dataset name to do")
    parser.add_argument('--data_root', type=str, default='./data', help="data root dir")
    parser.add_argument('--raw_root', type=str, default='./raw', help="raw root dir")
    parser.add_argument('--force', default=False, action='store_true', help="force to do task (otherwise use cached)")
    parser.add_argument('--random_seed', type=int, default=12345, help="random seed")

    # postprocessing
    args = parser.parse_args()
    args.data_root = Path(args.data_root)
    args.raw_root = Path(args.raw_root)

    return args


def print_timedelta(tdo):
    print(f"({'.'.join(str(tdo).split('.')[:-1])})")


def append_icontext(df_rows):
    df_rows['dto'] = pd.to_datetime(df_rows['stamp'], unit='s')
    (
        df_rows['year'],
        df_rows['month'],
        df_rows['day'],
        df_rows['weekday'],
        df_rows['yearday'],
        df_rows['week'],
        df_rows['hour'],
        df_rows['minutesecond'],
    ) = zip(*df_rows['dto'].map(lambda dto: (
        dto.year,
        dto.month,
        dto.day,
        dto.dayofweek,
        dto.dayofyear,
        dto.week,
        dto.hour,
        dto.minute * 60 + dto.second,
    )))

    # marker (relative)
    df_rows['marker'] = df_rows['stamp'].copy()
    df_rows['marker'] -= df_rows['marker'].min()
    df_rows['marker'] += 1
    df_rows['marker'] /= (df_rows['marker'].max() + 1)

    # year (relative)
    df_rows['year'] -= df_rows['year'].min()
    df_rows['year'] += 1
    df_rows['year'] /= (df_rows['year'].max() + 1)

    # month (1 ~ 12)
    df_rows['month_cos'] = np.cos(2 * np.pi * (df_rows['month'] - 1) / 12)
    df_rows['month_sin'] = np.sin(2 * np.pi * (df_rows['month'] - 1) / 12)
    df_rows['month'] /= 13

    # day (1 ~ 31)
    df_rows['day_cos'] = np.cos(2 * np.pi * (df_rows['day'] - 1) / 31)
    df_rows['day_sin'] = np.sin(2 * np.pi * (df_rows['day'] - 1) / 31)
    df_rows['day'] /= 32

    # weekday (0 ~ 6)
    df_rows['weekday_cos'] = np.cos(2 * np.pi * df_rows['weekday'] / 7)
    df_rows['weekday_sin'] = np.sin(2 * np.pi * df_rows['weekday'] / 7)
    df_rows['weekday'] = (df_rows['weekday'] + 1) / 8

    # yearday (1 ~ 366)
    df_rows['yearday_cos'] = np.cos(2 * np.pi * (df_rows['yearday'] - 1) / 366)
    df_rows['yearday_sin'] = np.sin(2 * np.pi * (df_rows['yearday'] - 1) / 366)
    df_rows['yearday'] /= 367

    # week (1 ~ 52)
    df_rows['week_cos'] = np.cos(2 * np.pi * (df_rows['week'] - 1) / 52)
    df_rows['week_sin'] = np.sin(2 * np.pi * (df_rows['week'] - 1) / 52)
    df_rows['week'] /= 53

    # hour (0 ~ 23)
    df_rows['hour_cos'] = np.cos(2 * np.pi * df_rows['hour'] / 24)
    df_rows['hour_sin'] = np.sin(2 * np.pi * df_rows['hour'] / 24)
    df_rows['hour'] = (df_rows['hour'] + 1) / 25

    # minutesecond (0 ~ 3599)
    df_rows['minutesecond_cos'] = np.cos(2 * np.pi * df_rows['minutesecond'] / 3600)
    df_rows['minutesecond_sin'] = np.sin(2 * np.pi * df_rows['minutesecond'] / 3600)
    df_rows['minutesecond'] = (df_rows['minutesecond'] + 1) / 3601

    # drop and reorder
    df_rows = df_rows.drop(columns=['dto'])
    df_rows = df_rows[['uid', 'iid', 'stamp'] + ICONTEXT_COLUMNS]
    return df_rows


def do_general_preprocessing(args, df_rows):
    """
        Given `df_rows` with a right format, the rest will be done.

        Args:
            `args`: see `parse_args`.
            `df_rows`: a DataFrame with column of `(uid, iid, stamp, year, month, day, dayofweek, dayofyear, week)`.
    """
    print("do general preprocessing")

    data_dir = args.data_root / args.dname

    if USE_FILTER_OUT:

        # filter out tiny items
        print("- filter out tiny items")
        df_iid2ucount = df_rows.groupby('iid').size()
        survived_iids = df_iid2ucount.index[df_iid2ucount >= MIN_USER_COUNT_PER_ITEM]
        df_rows = df_rows[df_rows['iid'].isin(survived_iids)]

        # filter out tiny users
        print("- filter out tiny users")
        df_uid2icount = df_rows.groupby('uid').size()
        survived_uids = df_uid2icount.index[df_uid2icount >= MIN_ITEM_COUNT_PER_USER]
        df_rows = df_rows[df_rows['uid'].isin(survived_uids)]

    if USE_CUT_OUT:

        print("- cut out sequence")
        uid2urows = {}
        for row in tqdm(df_rows.values, desc="* organizing"):
            uid = row[0]
            if uid not in uid2urows:
                uid2urows[uid] = []
            uid2urows[uid].append(row)
        rows = []
        uids = list(uid2urows.keys())
        for uid in tqdm(uids, desc="* cutting"):
            urows = uid2urows[uid]
            urows = urows[-MAX_SEQUENCE_LENGTH:]
            rows.extend(urows)
        df_rows = pd.DataFrame(rows)
        df_rows.columns = ['uid', 'iid', 'stamp'] + ICONTEXT_COLUMNS

    print("- map uid -> uindex", end=' ', flush=True)
    check = dt.now()
    ss_uids = df_rows.groupby('uid').size().sort_values(ascending=False)  # type: ignore
    uids = list(ss_uids.index)
    uid2uindex = {uid: index for index, uid in enumerate(uids, start=1)}
    df_rows['uindex'] = df_rows['uid'].map(uid2uindex)
    df_rows = df_rows.drop(columns=['uid'])
    with open(data_dir / 'uid2uindex.pkl', 'wb') as fp:
        pickle.dump(uid2uindex, fp)
    print_timedelta(dt.now() - check)

    print("- map iid -> iindex", end=' ', flush=True)
    check = dt.now()
    ss_iids = df_rows.groupby('iid').size().sort_values(ascending=False)  # type: ignore
    iids = list(ss_iids.index)
    iid2iindex = {iid: index for index, iid in enumerate(iids, start=1)}
    df_rows['iindex'] = df_rows['iid'].map(iid2iindex)
    df_rows = df_rows.drop(columns=['iid'])
    with open(data_dir / 'iid2iindex.pkl', 'wb') as fp:
        pickle.dump(iid2iindex, fp)
    print_timedelta(dt.now() - check)

    print("- save df_rows with icontext", end=' ', flush=True)
    check = dt.now()
    if USE_ICONTEXT_PCA:
        X = df_rows[ICONTEXT_COLUMNS].apply(tuple, axis=1)
        X = np.array([x for x in X])
        pca = PCA(n_components=ICONTEXT_PCA_DIM)
        X = pca.fit_transform(X)
        df_rows['icontext'] = [tuple(x) for x in X]
    else:
        df_rows['icontext'] = df_rows[ICONTEXT_COLUMNS].apply(tuple, axis=1)
    df_rows = df_rows.drop(columns=ICONTEXT_COLUMNS)
    df_rows = df_rows[['uindex', 'iindex', 'stamp', 'icontext']]
    df_rows.to_pickle(data_dir / 'df_rows.pkl')
    print_timedelta(dt.now() - check)

    print("- split train, valid, test")
    uindex2urows_train = {}
    uindex2urows_valid = {}
    uindex2urows_test = {}
    for uindex in tqdm(list(uid2uindex.values()), desc="* splitting"):
        df_urows = df_rows[df_rows['uindex'] == uindex]
        urows = list(df_urows[['iindex', 'stamp', 'icontext']].itertuples(index=False, name=None))
        if len(urows) < 3:
            uindex2urows_train[uindex] = urows
        else:
            uindex2urows_train[uindex] = urows[:-2]
            uindex2urows_valid[uindex] = urows[-2: -1]
            uindex2urows_test[uindex] = urows[-1:]

    print("- save splits", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'uindex2urows_train.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_train, fp)
    with open(data_dir / 'uindex2urows_valid.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_valid, fp)
    with open(data_dir / 'uindex2urows_test.pkl', 'wb') as fp:
        pickle.dump(uindex2urows_test, fp)
    print_timedelta(dt.now() - check)


def do_general_random_negative_sampling(args):
    """
        The `ns_random.pkl` created here is a dict with `uindex` as a key and a list of `iindex` as a value.

        `ns_random` = {dict of `uindex` -> [list of `iindex`]}.
    """
    print("do general random negative sampling")

    print("- init dirs")
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load materials", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'df_rows.pkl', 'rb') as fp:
        df_rows = pickle.load(fp)
    with open(data_dir / 'uid2uindex.pkl', 'rb') as fp:
        uid2uindex = pickle.load(fp)
        num_users = len(uid2uindex)
    with open(data_dir / 'iid2iindex.pkl', 'rb') as fp:
        iid2iindex = pickle.load(fp)
        num_items = len(iid2iindex)
    print_timedelta(dt.now() - check)

    print("- sample random negatives")
    ns = {}
    rng = Random(args.random_seed)
    for uindex in tqdm(list(range(1, num_users + 1)), desc="* sampling"):
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = set()
        for _ in range(NUM_NEGATIVE_SAMPLES):
            while True:
                iindex = rng.randint(1, num_items)
                if iindex in seen_iindices:
                    continue
                if iindex in sampled_iindices:
                    continue
                break
            sampled_iindices.add(iindex)
        ns[uindex] = list(sampled_iindices)

    print("- save sampled random nagetives", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'ns_random.pkl', 'wb') as fp:
        pickle.dump(ns, fp)
    print_timedelta(dt.now() - check)


def do_general_popular_negative_sampling(args):
    """
        The `ns_popular.pkl` created here is a dict with `uindex` as a key and a list of `iindex` as a value.

        `ns_popular` = {dict of `uindex` -> [list of `iindex`]}.
    """
    print("do general popular negative sampling")

    print("- init dirs")
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load materials", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'df_rows.pkl', 'rb') as fp:
        df_rows = pickle.load(fp)
    with open(data_dir / 'uid2uindex.pkl', 'rb') as fp:
        uid2uindex = pickle.load(fp)
        num_users = len(uid2uindex)
    print_timedelta(dt.now() - check)

    print("- reorder items")
    reordered_iindices = list(df_rows.groupby('iindex').size().sort_values(ascending=False).index)

    print("- sample popular negatives")
    ns = {}
    for uindex in tqdm(list(range(1, num_users + 1)), desc="* sampling"):
        seen_iindices = set(df_rows[df_rows['uindex'] == uindex]['iindex'])
        sampled_iindices = []
        for iindex in reordered_iindices:
            if len(sampled_iindices) == NUM_NEGATIVE_SAMPLES:
                break
            if iindex in seen_iindices:
                continue
            sampled_iindices.append(iindex)
        ns[uindex] = sampled_iindices

    print("- save sampled popular nagetives", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'ns_popular.pkl', 'wb') as fp:
        pickle.dump(ns, fp)
    print_timedelta(dt.now() - check)


def do_create_ifeature_matrix(args):
    """
        Uses `iid2feature` and `iid2iindex` to create `ifeatures` matrix.

        0th row has 0-vector.

        Args:
            `args`: see `parse_args`.
    """
    print("do create ifeatures matrix")

    print("- init dirs")
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load materials", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'iid2iindex.pkl', 'rb') as fp:
        iid2iindex = pickle.load(fp)
        iindex2iid = {iindex: iid for iid, iindex in iid2iindex.items()}
    with open(data_dir / 'iid2ifeature.pkl', 'rb') as fp:
        iid2ifeature = pickle.load(fp)
    print_timedelta(dt.now() - check)

    print("- create ifeatures matrix", end=' ', flush=True)
    check = dt.now()
    ifeatures = []
    for iindex in range(1, len(iid2iindex) + 1):
        iid = iindex2iid[iindex]
        ifeature = iid2ifeature[iid]
        ifeatures.append(ifeature)
    ifeature_dim = len(ifeatures[0])
    ifeatures = [np.zeros(ifeature_dim)] + ifeatures
    ifeatures = np.array(ifeatures)
    print_timedelta(dt.now() - check)

    print("- save ifeatures matrix", end=' ', flush=True)
    check = dt.now()
    with open(data_dir / 'ifeatures.pkl', 'wb') as fp:
        pickle.dump(ifeatures, fp)
    print_timedelta(dt.now() - check)


def preprocess_carca(args, ifeature_fname, rows_fname):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / 'CARCA'
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load ifeature data", end=' ', flush=True)
    check = dt.now()
    if not (data_dir / 'iid2ifeature.pkl').is_file():
        iid2ifeature = {}
        with open(raw_dir / ifeature_fname, 'rb') as fp:
            index2ifeature = pickle.load(fp)
            for iid, ifeature in enumerate(index2ifeature, start=1):
                iid = int(iid)
                iid2ifeature[iid] = ifeature
        with open(data_dir / 'iid2ifeature.pkl', 'wb') as fp:
            pickle.dump(iid2ifeature, fp)
    print_timedelta(dt.now() - check)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, dtype={
            0: int,
            1: int,
            2: int,
        }, delim_whitespace=True, header=None)
        df_rows.columns = ['uid', 'iid', 'stamp']
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)
    do_create_ifeature_matrix(args)

    print("done")
    print()


def task_prepare_fashion(args):
    preprocess_carca(
        args,
        ifeature_fname='Fashion_imgs.dat',
        rows_fname='Fashion_cxt.txt'
    )


def task_prepare_beauty(args):
    preprocess_carca(
        args,
        ifeature_fname='Beauty_feat_cat.dat',
        rows_fname='Beauty_cxt.txt'
    )


def task_prepare_men(args):
    preprocess_carca(
        args,
        ifeature_fname='Men_imgs.dat',
        rows_fname='Men_cxt.txt'
    )


def task_prepare_game(args):
    preprocess_carca(
        args,
        ifeature_fname='Video_Games_feat.dat',
        rows_fname='Video_Games_cxt.txt'
    )


def task_prepare_ml1m(args):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / args.dname
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    rows_fname = 'ratings.dat'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, dtype={
            0: int,
            1: int,
            2: int,
            3: int,
        }, sep='::', header=None, engine='python')
        df_rows.columns = ['uid', 'iid', 'rating', 'stamp']
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    if not (data_dir / 'iid2ifeature.pkl').is_file():

        print("- load ifeature data", end=' ', flush=True)
        check = dt.now()

        # load data
        df_item = pd.read_csv(raw_dir / 'movies.dat', sep='::', header=None, engine='python', encoding='latin-1')

        # check possible genres
        genreset = set()
        for iid, _, genres_str in df_item.itertuples(index=False, name=None):
            for genre in genres_str.split('|'):
                genreset.add(genre)

        # assign genre id
        genre2gid = {}  # type: ignore
        for genre in genreset:
            genre2gid[genre] = len(genre2gid)

        # create iid2ifeature
        iid2ifeature = {}
        for iid, _, genres_str in df_item.itertuples(index=False, name=None):
            ifeature = [0] * len(genreset)
            for genre in genres_str.split('|'):
                gid = genre2gid[genre]
                ifeature[gid] = 1
            iid2ifeature[iid] = tuple(ifeature)

        # save
        with open(data_dir / 'genre2gid.json', 'w') as fp:
            json.dump(genre2gid, fp)
        with open(data_dir / 'iid2ifeature.pkl', 'wb') as fp:
            pickle.dump(iid2ifeature, fp)

        print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    check = dt.now()
    df_rows = df_rows[df_rows['rating'] >= ML1M_MIN_RATING]
    df_rows = df_rows.drop(columns=['rating'])
    df_rows = df_rows[['uid', 'iid', 'stamp']]
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)
    do_create_ifeature_matrix(args)

    print("done")
    print()


def task_prepare_ml20m(args):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / args.dname
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    rows_fname = 'ratings.csv'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, dtype={
            'userId': int,
            'movieId': int,
            'rating': float,
            'timestamp': int,
        }, sep=',', header=0, engine='python')
        df_rows.columns = ['uid', 'iid', 'rating', 'stamp']
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    if not (data_dir / 'iid2ifeature.pkl').is_file():

        print("- load ifeature data", end=' ', flush=True)
        check = dt.now()

        # load data
        df_item = pd.read_csv(raw_dir / 'movies.csv', sep=',', header=0, engine='python')

        # check possible genres
        genreset = set()
        for iid, _, genres_str in df_item.itertuples(index=False, name=None):
            if genres_str == '(no genres listed)':
                continue
            for genre in genres_str.split('|'):
                genreset.add(genre)

        # assign genre id
        genre2gid = {}  # type: ignore
        for genre in genreset:
            genre2gid[genre] = len(genre2gid)

        # create iid2ifeature
        iid2ifeature = {}
        for iid, _, genres_str in df_item.itertuples(index=False, name=None):
            ifeature = [0] * len(genreset)
            if genres_str != '(no genres listed)':
                for genre in genres_str.split('|'):
                    gid = genre2gid[genre]
                    ifeature[gid] = 1
            iid2ifeature[iid] = tuple(ifeature)

        # save
        with open(data_dir / 'genre2gid.json', 'w') as fp:
            json.dump(genre2gid, fp)
        with open(data_dir / 'iid2ifeature.pkl', 'wb') as fp:
            pickle.dump(iid2ifeature, fp)

        print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    check = dt.now()
    df_rows = df_rows[df_rows['rating'] >= ML20M_MIN_RATING]
    df_rows = df_rows.drop(columns=['rating'])
    df_rows = df_rows[['uid', 'iid', 'stamp']]
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)
    do_create_ifeature_matrix(args)

    print("done")
    print()


def task_prepare_dressipi(args):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / args.dname
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    rows_fnames = (
        'train_sessions.csv',
        'test_full_sessions.csv',
    )
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        dfs = []
        for rows_fname in rows_fnames:
            df = pd.read_csv(raw_dir / rows_fname, dtype={
                'session_id': int,
                'item_id': int,
                'date': str,
            }, sep=',', header=0, engine='python')
            dfs.append(df)
        df_rows = pd.concat(dfs, ignore_index=True)
        df_rows.columns = ['uid', 'iid', 'date']
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    if not (data_dir / 'iid2ifeature.pkl').is_file():

        print("- load ifeature data", end=' ', flush=True)
        check = dt.now()

        # load data
        df_item = pd.read_csv(raw_dir / 'item_features.csv', sep=',', header=0, engine='python')

        # create category info
        df_item['category'] = 1000 * df_item['feature_category_id'] + df_item['feature_value_id']
        df_item = df_item.drop(columns=['feature_category_id', 'feature_value_id'])
        ss_category_count = df_item.groupby('category').size()

        # assign category id
        categoryset = list(ss_category_count[ss_category_count > 1].index)  # type: ignore
        category2cid = {}  # type: ignore
        for category in categoryset:
            category2cid[category] = len(category2cid)

        # create iid2ifeature
        iid2ifeature = {}
        for iid, category in df_item.itertuples(index=False, name=None):
            if category not in categoryset:
                continue
            cid = category2cid[category]
            if iid in iid2ifeature:
                iid2ifeature[iid][cid] = 1
            else:
                ifeature = [0] * len(categoryset)
                ifeature[cid] = 1
                iid2ifeature[iid] = ifeature
        for iid in list(iid2ifeature.keys()):
            iid2ifeature[iid] = tuple(iid2ifeature[iid])

        # save
        with open(data_dir / 'category2cid.json', 'w') as fp:
            json.dump(category2cid, fp)
        with open(data_dir / 'iid2ifeature.pkl', 'wb') as fp:
            pickle.dump(iid2ifeature, fp)

        print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    check = dt.now()
    df_rows['stamp'] = pd.to_datetime(df_rows['date'], format='%Y-%m-%d %H:%M:%S.%f').astype(int) / 1e+9
    df_rows = df_rows.astype({'stamp': int})
    df_rows = df_rows.drop(columns=['date'])
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows = df_rows[['uid', 'iid', 'stamp']]
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)
    do_create_ifeature_matrix(args)

    print("done")
    print()


def task_prepare_steam(args):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / args.dname
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    rows_fname = 'steam_reviews.json'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        rows = []
        with open(raw_dir / rows_fname, 'r') as fp:
            raw = fp.read().strip()
            lines = raw.split('\n')
            for line in tqdm(lines, desc="* loading"):
                line = line.strip()
                if not line:
                    continue
                one = eval(line)
                uid = one['username']
                iid = one['product_id']
                dto = dt.strptime(one['date'], '%Y-%m-%d')
                stamp = int(dto.timestamp())
                rows.append((uid, iid, stamp))
        df_rows = pd.DataFrame(rows, columns=['uid', 'iid', 'stamp'])
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)

    print("done")
    print()


def task_prepare_retailrocket(args):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / args.dname
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    rows_fname = 'events.csv'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, sep=',', header=0, engine='python')
        df_rows.columns = ['stampframe', 'uid', 'event', 'iid', 'tid']
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    df_rows = df_rows[df_rows['event'] == 'view']
    df_rows['stamp'] = df_rows['stampframe'] // 1000
    df_rows = df_rows.drop(columns=['event', 'tid', 'stampframe'])
    df_rows = df_rows[['uid', 'iid', 'stamp']]
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)

    print("done")
    print()


def task_prepare_diginetica(args):
    print(f"task: prepare {args.dname}")

    print("- init dirs")
    raw_dir = args.raw_root / args.dname
    data_dir = args.data_root / args.dname
    data_dir.mkdir(parents=True, exist_ok=True)

    print("- load log data", end=' ', flush=True)
    check = dt.now()
    fname = f'df_{args.dname}.pq'
    rows_fname = 'dataset/train-item-views.csv'
    if not args.force and (raw_dir / fname).is_file():
        df_rows = pd.read_parquet(raw_dir / fname)
    else:
        df_rows = pd.read_csv(raw_dir / rows_fname, dtype={
            'sessionId': int,
            'userId': str,
            'itemId': int,
            'timeframe': int,
            'eventdate': str,
        }, sep=';', header=0, engine='python')
        df_rows.columns = ['sid', 'uid', 'iid', 'frame', 'date']
        df_rows.to_parquet(raw_dir / fname)
    print_timedelta(dt.now() - check)

    print("- make raw df", end=' ', flush=True)
    df_rows['stamp'] = pd.to_datetime(df_rows['date'], format='%Y-%m-%d').astype(int) / 1e+9
    df_rows = df_rows.astype({'stamp': int})
    df_rows['stamp'] = df_rows['stamp'] + df_rows['frame']
    df_rows = df_rows.drop(columns=['uid', 'frame', 'date'])
    df_rows = df_rows.rename(columns={'sid': 'uid'})
    df_rows = df_rows[['uid', 'iid', 'stamp']]
    df_rows = df_rows.sort_values('stamp', ascending=True)
    df_rows.to_parquet(data_dir / 'df_rows_raw.pq')
    print_timedelta(dt.now() - check)

    print("- append icontext", end=' ', flush=True)
    check = dt.now()
    df_rows = append_icontext(df_rows)
    print_timedelta(dt.now() - check)

    do_general_preprocessing(args, df_rows)
    do_general_random_negative_sampling(args)
    do_general_popular_negative_sampling(args)

    print("done")
    print()


def task_count_stats(args):
    print("task: count stats")

    # init
    lines = []

    # collect dataset names
    dnames = []
    for cand in args.data_root.iterdir():
        if not cand.is_dir():
            continue
        dnames.append(cand.name)

    # print header
    message = '\t'.join([
        "dname",
        "#user",
        "#item",
        "#row",
        "density",
        "ic_mean",
        "ic_25",
        "ic_50",
        "ic_75",
        "ic_95",
        "ic_99",
    ])
    lines.append(message)
    print(message)

    # print rows
    for dname in dnames:
        data_dir = args.data_root / dname

        # load data
        with open(data_dir / 'uid2uindex.pkl', 'rb') as fp:
            uid2uindex = pickle.load(fp)
        with open(data_dir / 'iid2iindex.pkl', 'rb') as fp:
            iid2iindex = pickle.load(fp)
        with open(data_dir / 'df_rows.pkl', 'rb') as fp:
            df_rows = pickle.load(fp)

        # get item count per user
        icounts = df_rows.groupby('uindex').size().to_numpy()  # allow duplicates! not 'count'

        # get density
        num_users = len(uid2uindex)
        num_items = len(iid2iindex)
        num_rows = len(df_rows)
        density = num_rows / num_users / num_items

        # report
        message = '\t'.join([
            dname,
            str(num_users),
            str(num_items),
            str(num_rows),
            f"{100 * density:.04f}%",
            str(icounts.mean()),
            str(int(np.percentile(icounts, 25))),
            str(int(np.percentile(icounts, 50))),
            str(int(np.percentile(icounts, 75))),
            str(int(np.percentile(icounts, 95))),
            str(int(np.percentile(icounts, 99))),
        ])
        lines.append(message)
        print(message)

    # save to file too
    with open(args.data_root / 'dstats.tsv', 'w') as fp:
        fp.write('\n'.join(lines))


if __name__ == '__main__':
    args = parse_args()
    if args.task == 'prepare':
        globals()[f'task_{args.task}_{args.dname}'](args)
    else:
        globals()[f'task_{args.task}'](args)
