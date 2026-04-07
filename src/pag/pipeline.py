import pytorch_lightning as pl
import torch as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import pandas as pd

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
        self.post_freeze_lr = hyperparams.get('post-freeze-lr')
        if self.post_freeze_lr is None:
            self.post_freeze_lr = self.lr / 10.0
        self.post_freeze_plateau_factor = hyperparams.get('post-freeze-plateau-factor', 0.5)
        self.post_freeze_plateau_patience = hyperparams.get('post-freeze-plateau-patience', 10)
        self.post_freeze_min_lr = hyperparams.get('post-freeze-min-lr', 1e-6)
        self.weight_decay = hyperparams.get('weight-decay', 1e-4)
        self.mc_sample_size = hyperparams.get('mc-sample-size', 128)
        self.print_loss_every = hyperparams.get('print-loss-every', 10)
        self.validation_sample_count = hyperparams.get('validation-sample-count', 10000)
        self.validation_depth = hyperparams.get('validation-depth', 3)
        self.print_prob_table_every = hyperparams.get('print-prob-table-every', 1)
        self.lambda_dag = hyperparams.get('lambda-dag', 1.0)
        self.lambda_l1 = hyperparams.get('lambda-l1', 1e-3)
        self.freeze_dag_when_stable = hyperparams.get('freeze-dag-when-stable', False)
        self.soft_reset_optimizer_when_stable = hyperparams.get('soft-reset-optimizer-when-stable', False)
        self.soft_reset_max_count = hyperparams.get('soft-reset-max-count', 0)
        self.freeze_dag_min_epochs = hyperparams.get('freeze-dag-min-epochs', 20)
        self.freeze_dag_window = hyperparams.get('freeze-dag-window', 5)
        self.freeze_dag_adj_tol = hyperparams.get('freeze-dag-adj-tol', 1e-3)
        self.freeze_dag_penalty_threshold = hyperparams.get('freeze-dag-penalty-threshold', 1e-4)
        self.ground_truth_prob_table = ground_truth_prob_table
        self.best_val_sample_kl = None
        self.best_val_sample_epoch = None
        self.best_sampled_table = None
        self.best_adjacency = None
        self.dag_frozen = False
        self.dag_frozen_epoch = None
        self._recent_adjacencies = []
        self.soft_reset_count = 0
        self._graph_was_stable_last_epoch = False
        self._post_freeze_optimizer_reset = False
        self._post_freeze_scheduler = None

        self.model = PAGModel(
            num_variables=train_data.shape[1],
            graph=self.graph,
            mask_structure=hyperparams.get('mask-structure', 'learnable'),
            latent_dim=hyperparams.get('latent-dim', 8),
            embed_dim=hyperparams.get('embed-dim', 64),
            num_heads=hyperparams.get('num-heads', 4),
            num_layers=hyperparams.get('num-layers', 1),
            ffn_hidden_dim=hyperparams.get('ffn-hidden-dim', 128),
            latent_mlp_hidden_dim=hyperparams.get('latent-mlp-hidden-dim', 128),
            latent_mlp_layers=hyperparams.get('latent-mlp-layers', 2),
            head_hidden_dim=hyperparams.get('head-hidden-dim', 64),
            dagma_s=hyperparams.get('dagma-s', 2.0),
            init_edge_logit=hyperparams.get('mask-init-logit', -0.275),
            mask_init_std=hyperparams.get('mask-init-std', 1.0),
            adjacency_normalization=hyperparams.get('adjacency-normalization', 'none'),
            gate_floor=hyperparams.get('gate-floor', 0.05),
            gate_renorm_eps=hyperparams.get('gate-renorm-eps', 1e-6),
            dropout=hyperparams.get('dropout', 0.0),
            latent_prior=hyperparams.get('latent-prior', 'normal'),
            residual=hyperparams.get('residual', True),
        )
        self.initial_adjacency = self.model.get_observed_adjacency().detach().cpu().clone()

    def configure_optimizers(self):
        optimizer = T.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._post_freeze_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.post_freeze_plateau_factor,
            patience=self.post_freeze_plateau_patience,
            min_lr=self.post_freeze_min_lr,
        )
        return optimizer

    def train_dataloader(self):
        return DataLoader(BinaryTableDataset(self.train_data), batch_size=self.batch_size, shuffle=True, drop_last=False)

    def val_dataloader(self):
        if self.val_data is None:
            return None
        return DataLoader(BinaryTableDataset(self.val_data), batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _reset_optimizer_for_post_freeze_phase(self):
        if self._post_freeze_optimizer_reset:
            return
        trainer = getattr(self, '_trainer', None)
        if trainer is None or not getattr(trainer, 'optimizers', None):
            return

        for optimizer in trainer.optimizers:
            optimizer.state.clear()
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.post_freeze_lr

        self._post_freeze_optimizer_reset = True
        rank_zero_info(
            "reset optimizer state and switched post-freeze lr to {:.6g}".format(self.post_freeze_lr)
        )

    def _reset_optimizer_state_without_lr_change(self):
        trainer = getattr(self, '_trainer', None)
        if trainer is None or not getattr(trainer, 'optimizers', None):
            return False

        current_lrs = []
        for optimizer in trainer.optimizers:
            optimizer.state.clear()
            current_lrs.extend(param_group['lr'] for param_group in optimizer.param_groups)

        rank_zero_info(
            "soft-reset optimizer state while keeping lr at {}".format(
                ", ".join("{:.6g}".format(lr) for lr in current_lrs)
            )
        )
        return True

    def _step_post_freeze_scheduler(self, sampled_kl):
        if not self.dag_frozen or self._post_freeze_scheduler is None:
            return

        optimizer = self._post_freeze_scheduler.optimizer
        old_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        self._post_freeze_scheduler.step(float(sampled_kl))
        new_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        if any(abs(new_lr - old_lr) > 0.0 for old_lr, new_lr in zip(old_lrs, new_lrs)):
            rank_zero_info(
                "post-freeze plateau scheduler reduced lr from {} to {} after val_sample_kl={:.6f}".format(
                    ", ".join("{:.6g}".format(lr) for lr in old_lrs),
                    ", ".join("{:.6g}".format(lr) for lr in new_lrs),
                    float(sampled_kl),
                )
            )

    def _log_adjacency(self, stage):
        adjacency = self.model.get_observed_adjacency()
        edge_names = [
            ('x_to_y', adjacency[0, 1]),
            ('y_to_x', adjacency[1, 0]),
            ('y_to_z', adjacency[1, 2]),
            ('z_to_y', adjacency[2, 1]),
        ]
        for name, value in edge_names:
            self.log(
                '{}_{}'.format(stage, name),
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

    def _can_freeze_dag(self):
        return (
            self.freeze_dag_when_stable
            and not self.dag_frozen
            and self.model.mask_structure == 'learnable'
            and getattr(self.model.mask_module, 'edge_logits', None) is not None
        )

    def _update_adjacency_history(self, adjacency):
        self._recent_adjacencies.append(adjacency.detach().cpu().clone())
        if len(self._recent_adjacencies) > self.freeze_dag_window:
            self._recent_adjacencies = self._recent_adjacencies[-self.freeze_dag_window:]

    def _adjacency_is_stable(self):
        if len(self._recent_adjacencies) < self.freeze_dag_window:
            return False
        max_diff = 0.0
        for idx in range(1, len(self._recent_adjacencies)):
            diff = (self._recent_adjacencies[idx] - self._recent_adjacencies[idx - 1]).abs().max().item()
            max_diff = max(max_diff, diff)
        return max_diff <= self.freeze_dag_adj_tol

    def _graph_is_stably_eligible(self, dag_penalty, epoch_idx):
        if epoch_idx + 1 < self.freeze_dag_min_epochs:
            return False
        if float(dag_penalty.detach().cpu()) > self.freeze_dag_penalty_threshold:
            return False
        return self._adjacency_is_stable()

    def _can_soft_reset_optimizer(self):
        return (
            self.soft_reset_optimizer_when_stable
            and not self.freeze_dag_when_stable
            and not self.dag_frozen
            and self.soft_reset_count < self.soft_reset_max_count
            and self.model.mask_structure == 'learnable'
            and getattr(self.model.mask_module, 'edge_logits', None) is not None
        )

    def _maybe_soft_reset_optimizer(self, stable_now):
        did_reset = False
        if self._can_soft_reset_optimizer() and stable_now and not self._graph_was_stable_last_epoch:
            did_reset = self._reset_optimizer_state_without_lr_change()
            if did_reset:
                self.soft_reset_count += 1
                rank_zero_info(
                    "soft stable-graph optimizer reset {}/{}".format(
                        self.soft_reset_count,
                        self.soft_reset_max_count,
                    )
                )
        self._graph_was_stable_last_epoch = stable_now
        return did_reset

    def _maybe_freeze_dag(self, adjacency, dag_penalty, epoch_idx=None, history_already_updated=False):
        if not self._can_freeze_dag():
            return

        if epoch_idx is None:
            epoch_idx = int(self.current_epoch)
        if not history_already_updated:
            self._update_adjacency_history(adjacency)
        if not self._graph_is_stably_eligible(dag_penalty, epoch_idx):
            return

        freeze_order = self.model._build_confidence_order()
        self.model.mask_module.freeze_to_order(freeze_order)
        self.model.mask_module.edge_logits.requires_grad_(False)
        self.dag_frozen = True
        self.dag_frozen_epoch = int(epoch_idx)
        self._reset_optimizer_for_post_freeze_phase()
        rank_zero_info(
            "freezing dag at epoch={} with order={} because adjacency stabilized and dag_penalty={:.6f}".format(
                self.dag_frozen_epoch,
                ", ".join(self.variable_names[idx] for idx in freeze_order),
                float(dag_penalty.detach().cpu()),
            )
        )

    def _shared_step(self, batch, stage):
        log_prob = self.model.marginal_log_prob(batch.float(), self.mc_sample_size)
        nll = -log_prob.mean()
        dag_penalty = self.model.dag_penalty()
        l1_penalty = self.model.l1_penalty()
        loss = nll + self.lambda_dag * dag_penalty + self.lambda_l1 * l1_penalty

        is_train = stage == 'train'
        self.log('{}_loss'.format(stage), loss, prog_bar=(stage != 'train'), on_step=is_train, on_epoch=True)
        self.log('{}_mean_log_prob'.format(stage), log_prob.mean(), prog_bar=False, on_step=is_train, on_epoch=True)
        self.log('{}_nll'.format(stage), nll, prog_bar=False, on_step=is_train, on_epoch=True)
        self.log('{}_dag_penalty'.format(stage), dag_penalty, prog_bar=False, on_step=is_train, on_epoch=True)
        self.log('{}_l1_penalty'.format(stage), l1_penalty, prog_bar=False, on_step=is_train, on_epoch=True)
        self.log('dag_frozen', float(self.dag_frozen), prog_bar=False, on_step=False, on_epoch=True)
        self._log_adjacency(stage)
        return {
            'loss': loss,
            'mean_log_prob': log_prob.mean(),
            'nll': nll,
            'dag_penalty': dag_penalty,
            'l1_penalty': l1_penalty,
        }

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, 'train')

        if self.print_loss_every and batch_idx % self.print_loss_every == 0:
            rank_zero_info(
                "epoch={} batch={} train_loss={:.6f} mean_log_prob={:.6f} nll={:.6f} dag_penalty={:.6f} l1_penalty={:.6f}".format(
                    int(self.current_epoch),
                    int(batch_idx),
                    float(metrics['loss'].detach().cpu()),
                    float(metrics['mean_log_prob'].detach().cpu()),
                    float(metrics['nll'].detach().cpu()),
                    float(metrics['dag_penalty'].detach().cpu()),
                    float(metrics['l1_penalty'].detach().cpu()),
                )
            )
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch, 'val')
        return metrics['loss']

    def on_validation_epoch_end(self):
        if self._can_freeze_dag() or self._can_soft_reset_optimizer() or self.dag_frozen:
            adjacency = self.model.get_observed_adjacency().detach().cpu()
            dag_penalty = self.model.dag_penalty()
            epoch_idx = int(self.current_epoch)
            self._update_adjacency_history(adjacency)
            stable_now = self._graph_is_stably_eligible(dag_penalty, epoch_idx)
            self._maybe_soft_reset_optimizer(stable_now)
            self._maybe_freeze_dag(adjacency, dag_penalty, epoch_idx=epoch_idx, history_already_updated=True)

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
            rank_zero_info("validation sampling order={}".format(", ".join(order_names)))
            adjacency = self.model.get_observed_adjacency().detach().cpu()
            rank_zero_info("validation adjacency:\n{}".format(adjacency))
            if self.dag_frozen:
                rank_zero_info("adjacent is fixed now")

        sampled = self.model.sample_chain(
            sample_count=self.validation_sample_count,
            depth=self.validation_depth,
            device=self.device,
        )
        sampled_table = binary_table_probability_table(sampled, self.variable_names)
        sampled_kl = probability_table_kl(self.ground_truth_prob_table, sampled_table)
        adjacency = self.model.get_observed_adjacency().detach().cpu()

        if self.best_val_sample_kl is None or sampled_kl < self.best_val_sample_kl:
            self.best_val_sample_kl = float(sampled_kl)
            self.best_val_sample_epoch = int(self.current_epoch)
            self.best_sampled_table = sampled_table.copy()
            self.best_adjacency = adjacency.clone()

        self.log('val_sample_kl', sampled_kl, prog_bar=True, on_step=False, on_epoch=True)
        post_freeze_sample_kl = sampled_kl if self.dag_frozen else float('inf')
        self.log('val_post_freeze_sample_kl', post_freeze_sample_kl, prog_bar=False, on_step=False, on_epoch=True)
        self._step_post_freeze_scheduler(sampled_kl)
        if should_print:
            rank_zero_info("validation true probability table:\n{}".format(
                self.ground_truth_prob_table.to_string(index=False)))
            rank_zero_info("validation sampled probability table:\n{}".format(sampled_table.to_string(index=False)))
            rank_zero_info("validation sampled KL={:.6f}".format(sampled_kl))

    def get_best_kl_summary(self):
        if self.best_val_sample_kl is None:
            return None

        true_table = self.ground_truth_prob_table
        if isinstance(true_table, pd.DataFrame):
            true_table_records = true_table.to_dict(orient='records')
        else:
            true_table_records = true_table

        return {
            'best_kl': float(self.best_val_sample_kl),
            'best_kl_epoch': int(self.best_val_sample_epoch),
            'estimated_table': self.best_sampled_table.to_dict(orient='records'),
            'true_table': true_table_records,
            'adjacency_matrix': self.best_adjacency.tolist(),
        }
