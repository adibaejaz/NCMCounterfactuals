import pytorch_lightning as pl
import torch as T
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from .data import BinaryTableDataset, binary_table_probability_table, probability_table_kl
from .model import PAGModel


class PAGPipeline(pl.LightningModule):
    def __init__(
        self,
        train_data,
        val_data=None,
        variable_names=None,
        ground_truth_prob_table=None,
        hyperparams=None,
    ):
        super().__init__()
        if hyperparams is None:
            hyperparams = {}

        self.save_hyperparameters(ignore=['train_data', 'val_data', 'ground_truth_prob_table'])

        self.train_data = train_data.float()
        self.val_data = None if val_data is None else val_data.float()
        self.variable_names = variable_names or ["v{}".format(i) for i in range(train_data.shape[1])]
        self.graph = hyperparams.get('graph', 'chain')

        if self.graph != 'chain':
            raise ValueError("Only graph='chain' is supported right now.")
        if train_data.shape[1] != 3:
            raise ValueError("The chain PAG pipeline requires exactly three variables: X, Y, Z.")

        normalized_names = [str(name).upper() for name in self.variable_names]
        if normalized_names != ['X', 'Y', 'Z']:
            raise ValueError("The chain PAG pipeline requires variable names ['X', 'Y', 'Z'].")

        self.batch_size = hyperparams.get('batch-size', 256)
        self.lr = hyperparams.get('lr', 1e-3)
        self.weight_decay = hyperparams.get('weight-decay', 1e-4)
        self.mc_sample_size = hyperparams.get('mc-sample-size', 128)
        self.print_loss_every = hyperparams.get('print-loss-every', 10)
        self.validation_sample_count = hyperparams.get('validation-sample-count', 10000)
        self.validation_depth = hyperparams.get('validation-depth', 3)
        self.print_prob_table_every = hyperparams.get('print-prob-table-every', 1)
        self.ground_truth_prob_table = ground_truth_prob_table

        self.model = PAGModel(
            num_variables=train_data.shape[1],
            graph=self.graph,
            latent_dim=hyperparams.get('latent-dim', 8),
            embed_dim=hyperparams.get('embed-dim', 64),
            num_heads=hyperparams.get('num-heads', 4),
            num_layers=hyperparams.get('num-layers', 1),
            ffn_hidden_dim=hyperparams.get('ffn-hidden-dim', 128),
            latent_mlp_hidden_dim=hyperparams.get('latent-mlp-hidden-dim', 128),
            latent_mlp_layers=hyperparams.get('latent-mlp-layers', 2),
            head_hidden_dim=hyperparams.get('head-hidden-dim', 64),
            mask_mode=hyperparams.get('mask-mode', 'additive'),
            dropout=hyperparams.get('dropout', 0.0),
            latent_prior=hyperparams.get('latent-prior', 'normal'),
            residual=hyperparams.get('residual', True),
        )

    def configure_optimizers(self):
        optimizer = T.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return DataLoader(BinaryTableDataset(self.train_data), batch_size=self.batch_size, shuffle=True, drop_last=False)

    def val_dataloader(self):
        if self.val_data is None:
            return None
        return DataLoader(BinaryTableDataset(self.val_data), batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _shared_step(self, batch, stage):
        log_prob = self.model.marginal_log_prob(batch.float(), self.mc_sample_size)
        loss = -log_prob.mean()

        is_train = stage == 'train'
        self.log('{}_loss'.format(stage), loss, prog_bar=(stage != 'train'), on_step=is_train, on_epoch=True)
        self.log('{}_mean_log_prob'.format(stage), log_prob.mean(), prog_bar=False, on_step=is_train, on_epoch=True)
        return loss, log_prob.mean()

    def training_step(self, batch, batch_idx):
        loss, mean_log_prob = self._shared_step(batch, 'train')

        if self.print_loss_every and batch_idx % self.print_loss_every == 0:
            rank_zero_info(
                "epoch={} batch={} train_loss={:.6f} mean_log_prob={:.6f}".format(
                    int(self.current_epoch),
                    int(batch_idx),
                    float(loss.detach().cpu()),
                    float(mean_log_prob.detach().cpu()),
                )
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, 'val')
        return loss

    def on_validation_epoch_end(self):
        if self.ground_truth_prob_table is None or self.validation_sample_count <= 0:
            return
        if self.graph != 'chain':
            raise ValueError("Only graph='chain' is supported right now.")
        if self.validation_depth != 3:
            raise ValueError("The chain PAG validation sampler requires depth=3.")

        should_print = self.print_prob_table_every and (self.current_epoch % self.print_prob_table_every == 0)
        if should_print:
            update_order = self.model._build_confidence_order()
            order_names = [self.variable_names[idx] for idx in update_order]
            rank_zero_info("validation sampling order={}".format(" -> ".join(order_names)))

        sampled = self.model.sample_chain(
            sample_count=self.validation_sample_count,
            depth=self.validation_depth,
            device=self.device,
        )
        sampled_table = binary_table_probability_table(sampled, self.variable_names)
        sampled_kl = probability_table_kl(self.ground_truth_prob_table, sampled_table)

        self.log('val_sample_kl', sampled_kl, prog_bar=True, on_step=False, on_epoch=True)
        if should_print:
            rank_zero_info("validation sampled probability table:\n{}".format(sampled_table.to_string(index=False)))
            rank_zero_info("validation sampled KL={:.6f}".format(sampled_kl))
