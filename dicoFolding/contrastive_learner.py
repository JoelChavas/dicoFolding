import torch
from dicoFolding.losses import NTXenLoss
from dicoFolding.models.densenet import DenseNet


class ContrastiveLearner(DenseNet):

    def __init__(self, config, mode, drop_rate):
        super(ContrastiveLearner, self).__init__(growth_rate=32,
                                                 block_config=(6, 12, 24, 16),
                                                 num_init_features=64,
                                                 mode=mode,
                                                 drop_rate=drop_rate)
        self.config = config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        return optimizer

    def nt_xen_loss(self, z_i, z_j):
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def training_step(self, train_batch, batch_idx):
        (inputs, filenames) = train_batch
        z_i = self.forward(inputs[:, 0, :])
        z_j = self.forward(inputs[:, 1, :])
        batch_loss, _, _ = self.nt_xen_loss(z_i, z_j)
        self.log('train_loss', float(batch_loss))
        return batch_loss

    def validation_step(self, val_batch, batch_idx):
        (inputs, filenames) = val_batch
        z_i = self.forward(inputs[:, 0, :])
        z_j = self.forward(inputs[:, 1, :])
        batch_loss, logits, target = self.nt_xen_loss(z_i, z_j)
        self.log('val_loss', float(batch_loss))
