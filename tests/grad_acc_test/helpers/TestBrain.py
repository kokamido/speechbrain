import sys

sys.path.insert(1, '../..')

import speechbrain as sb


class TestBrain(sb.Brain):
    def __init__(self, modules=None, opt_class=None, hparams=None, run_opts=None, checkpointer=None):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.loss = hparams['loss']
        self.losses = []

    def compute_objectives(self, predictions, batch, stage):
        _, labels = batch
        loss = self.loss(predictions, labels.to(self.device))
        if stage == sb.Stage.VALID:
            self.losses.append(float(loss))
        return loss

    def compute_forward(self, batch, stage):
        data, _ = batch
        return self.modules['model'](data.to(self.device)).squeeze()
