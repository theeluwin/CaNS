import json
import argparse
import multiprocessing

from pathlib import Path

from solvers import *  # noqa: F401,F403


# default
ROOT = Path(__file__).absolute().parents[0]
RUNS = 'runs'
default_config = {
    'envs': {
        'RUN_ROOT': str(ROOT / RUNS),
        'DATA_ROOT': str(ROOT / 'data'),
        'RAW_ROOT': str(ROOT / 'raw'),
        # 'CPU_COUNT': 4,
        'CPU_COUNT': multiprocessing.cpu_count() // 8,
        'GPU_COUNT': 1,
    },
    'solver': 'PopSolver',
    'dataset': 'ml1m',
    'dataloader': {
        'sequence_len': 100,  # depends on dataset
        'train_num_negatives': 100,
        'valid_num_negatives': 100,
        'random_cut_prob': 1.0,
        'replace_user_prob': 0.0,
        'replace_item_prob': 0.01,
        'clustering_method': 'iterative',  # ('iterative', 'kmeans', 'interval')
        'num_ccs': 10,  # number of context clusters (for CaPop)
        'ncs': 'none',  # negative context sampling ('none', 'random', 'hard')
        'ncs_ccs': 1,
        'nis': 'uniform',  # negative item sampling ('negative', 'uniform', 'popular')
        'nis_ccs': 1,
        'random_seed': 0,
    },
    'model': {
        'hidden_dim': 64,
        'num_layers': 1,
        'num_heads': 4,
        'dropout_prob': 0.1,
        'temperature': 0.1,
        'random_seed': 0,
    },
    'train': {
        'epoch': 600,  # E
        'every': 10,  # E / 20
        'patience': 100,  # 2 * E / 5
        'batch_size': 128,
        'optimizer': {
            'algorithm': 'adamw',
            'lr': 1e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.1,
            'amsgrad': False,
        },
        'scheduler': {
            'algorithm': None,
        },
    },
    'metric': {
        'ks_valid': [10],
        'ks_test': [1, 5, 10, 20, 50, 100],
        'pivot': 'NDCG@10',
    },
    'memo': "",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help="run to execute")
    return parser.parse_args()


def update_dict_diff(base, diff):
    for key, value in diff.items():
        if isinstance(value, dict) and value:
            partial = update_dict_diff(base.get(key, {}), value)
            base[key] = partial
        else:
            base[key] = diff[key]
    return base


if __name__ == '__main__':

    # args
    args = parse_args()

    # settle dirs
    run_root = ROOT / RUNS
    run_dir = run_root / args.name
    if not run_root.is_dir():
        raise Exception(f"You need to create a `{RUNS}` directory.")
    if not run_dir.is_dir():
        raise Exception("You need to create your run directory.")

    # check config file
    final_config_path = run_dir / 'config.json'
    if not final_config_path.is_file():
        raise Exception("You need to create a `config.json` in your run directory.")

    # get and update config
    config = dict(default_config)
    partial_names = args.name.split('/')
    for i in range(1, len(partial_names) + 1):
        partial_config_path = run_root / '/'.join(partial_names[:i]) / 'config.json'
        if partial_config_path.is_file():
            with open(partial_config_path, 'r') as fp:
                partial_config = json.load(fp)
                update_dict_diff(config, partial_config)

    # settle config
    config['name'] = args.name
    config['run_dir'] = str(run_dir)

    # lock config
    with open(run_dir / 'config-lock.json', 'w') as fp:
        json.dump(config, fp, indent=4)

    # run
    solver_name = str(config['solver'])
    solver_class = globals()[solver_name]
    solver = solver_class(config)
    solver.solve()
