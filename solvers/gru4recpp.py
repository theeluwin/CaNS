import pickle

from shutil import copyfile
from pathlib import Path

from torch import topk as torch_topk

from models import GRU4RecPP

from .base import BaseLWPContrastiveSolver


__all__ = (
    'GRU4RecPPSolver',
)


class GRU4RecPPSolver(BaseLWPContrastiveSolver):

    # override
    def init_model(self) -> None:
        C = self.config
        CM = C['model']
        data_root = Path(C['envs']['DATA_ROOT'])

        # get num items
        with open(data_root / C['dataset'] / 'iid2iindex.pkl', 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.num_items = len(self.iid2iindex)

        # get ifeatures
        try:
            with open(data_root / C['dataset'] / 'ifeatures.pkl', 'rb') as fp:
                ifeatures = pickle.load(fp)
            ifeatures_dim = ifeatures.shape[1]
        except FileNotFoundError:
            ifeatures = None
            ifeatures_dim = 0

        # init model
        self.model = GRU4RecPP(
            num_items=self.num_items,
            ifeatures=ifeatures,  # type: ignore
            ifeature_dim=ifeatures_dim,
            icontext_dim=self.train_dataset.icontext_dim,  # type: ignore
            hidden_dim=CM['hidden_dim'],
            num_layers=CM['num_layers'],
            dropout_prob=CM['dropout_prob'],
            random_seed=CM['random_seed'],
        ).to(self.device)

    def calculate_forward(self, batch):

        # device
        profile_tokens = batch['profile_tokens'].to(self.device)  # b x L
        profile_icontexts = batch['profile_icontexts'].to(self.device)  # b x L x d_Ci
        extract_tokens = batch['extract_tokens'].to(self.device)  # b x C
        extract_icontexts = batch['extract_icontexts'].to(self.device)  # b x C x d_Ci

        # forward
        logits = self.model(
            profile_tokens,
            profile_icontexts,
            extract_tokens,
            extract_icontexts,
        )  # b x C

        return logits

    # override
    def calculate_loss(self, batch):

        # device
        label = batch['label'].to(self.device)  # b

        # forward
        logits = self.calculate_forward(batch)  # b x C
        logits = logits / self.config['model']['temperature']

        # loss
        loss = self.ce_losser(logits, label)

        return loss

    # override
    def calculate_rankers(self, batch):

        # forward
        logits = self.calculate_forward(batch)  # b x C

        # ranking
        _, rankers = torch_topk(logits, self.max_top_k, dim=1)

        return rankers

    # override
    def backup(self):
        copyfile('datasets.py', self.data_dir / 'datasets.py')
        copyfile('models/encoders/advanced.py', self.data_dir / 'encoder.py')
        copyfile('models/gru4recpp.py', self.data_dir / 'model.py')
        copyfile('solvers/gru4recpp.py', self.data_dir / 'solver.py')
