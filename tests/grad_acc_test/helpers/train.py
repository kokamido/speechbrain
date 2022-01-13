import os.path
import random
import sys

import numpy as np

sys.path.insert(1, '../..')

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from TestBrain import TestBrain
from TestClassifier import TestClassifier


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_loaders(hparams, seed):
    X, y = make_classification(hparams['dataset_samples_count'], hparams['dataset_features_count'],
                               shuffle=False, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(X[:, None, :], y, test_size=0.2, shuffle=True,
                                                        random_state=seed)

    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)),
                              batch_size=hparams['train_batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)),
                             batch_size=hparams['val_batch_size'], shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    seed_everything(1234)
    hparams_overrides_file, run_opts, _ = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    with open('configs/hparams.yaml') as fin:
        hparams = load_hyperpyyaml(fin)

    with open(hparams_overrides_file) as fin:
        hparams_overrides = load_hyperpyyaml(fin)
        for param in hparams_overrides:
            hparams[param] = hparams_overrides[param]

    train_loader, test_loader = get_loaders(hparams, 1234)

    modules = {'model': TestClassifier()}

    brain = TestBrain(modules, hparams['opt_class'], hparams, run_opts)

    brain.fit(hparams['epoch_counter'], train_loader, test_loader)

    with open(os.path.join('results_grad_acc', hparams['name']), 'w') as out:
        out.write('\n'.join(map(str, brain.losses)))
