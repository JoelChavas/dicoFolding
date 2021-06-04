import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)


class ContrastiveLearningModel:

    def __init__(self, net, loss,
                 loader_train, loader_val,
                 config, scheduler=None):
        """Inits model with neural net, loss and loaders

        Args:
            net: subclass of nn.Module
            loss: callable fn with args (y_pred, y_true)
            loader_train: pytorch DataLoader for training
            Loader_val: pytorch DataLoader for validation
            config: config object with hyperparameters
            scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("ContrastiveLearning")
        self.loss = loss
        self.model = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr,
                                          weight_decay=config.weight_decay)
        self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.metrics = {}

        self.model = DataParallel(self.model).to(self.device)

    def training(self):
        """Main training loop
        """
        log.info(self.loss)
        log.info(self.optimizer)

        start_epoch = self.config.start_epoch

        if start_epoch > 0:
            self.load_checkpoint(start_epoch)

        for epoch in range(start_epoch, self.config.nb_epochs):

            # Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, filenames) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                z_i = self.model(inputs[:, 0, :])
                z_j = self.model(inputs[:, 1, :])
                batch_loss, logits, target = self.loss(z_i, z_j)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
            pbar.close()

            # Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_values = {}
            with torch.no_grad():
                self.model.eval()
                for (inputs, filenames) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    z_i = self.model(inputs[:, 0, :])
                    z_j = self.model(inputs[:, 1, :])
                    batch_loss, logits, target = self.loss(z_i, z_j)
                    val_loss += float(batch_loss) / nb_batch
                    for name, metric in self.metrics.items():
                        if name not in val_values:
                            val_values[name] = 0
                        val_values[name] += metric(logits, target) / nb_batch
            pbar.close()

            metrics = "\t".join(["Validation {}: {:.4f}".format(m, v)
                                 for (m, v) in val_values.items()])
            log.info("Epoch [{}/{}] Training loss = {:.4f}\t"
                     "Validation loss = {:.4f}\t".format(
                         epoch + 1, self.config.nb_epochs,
                         training_loss, val_loss) + metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_saving_true(epoch):
                self.save_checkpoint(epoch)

    def is_saving_true(self, epoch):
        """Determines if checkpoint saving will be performed.

        Args:
            epoch: epoch corresponding to the checkpoint
        """
        return (epoch % self.config.nb_epochs_per_saving == 0
                or epoch == self.config.nb_epochs - 1) and epoch > 0

    def checkpoint_path(self, epoch):
        """Gives full path of checkpoint file.

        Args:
            epoch: epoch corresponding to the checkpoint
        """
        return os.path.join(
            self.config.checkpoint_dir,
            "{name}_epoch_{epoch}.pth". format(
                name=self.config.name,
                epoch=epoch))

    def save_checkpoint(self, epoch):
        """Saves checkpoint (both model and optimizer).

        Args:
            epoch: epoch corresponding to the checkpoint
        """
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()},
            self.checkpoint_path(epoch))

    def load_last_checkpoint(self):
        """
        """

    def load_checkpoint(self, epoch):
        """Loads checkpoint (both model and optimizer)

        Args:
            epoch: epoch corresponding to the checkpoint
        """
        checkpoint = None
        try:
            checkpoint = torch.load(
                self.checkpoint_path(epoch),
                map_location=lambda storage,
                loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(
                        checkpoint.state_dict())
                    self.logger.info(
                        'Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(
                            checkpoint["model"], strict=False)
                        self.logger.info(
                            'Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info(
                        'Model loading info: {}'.format(unexpected))
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except BaseException as e:
                raise ValueError(
                    'Error while loading the model\'s weights: %s' %
                    str(e))
