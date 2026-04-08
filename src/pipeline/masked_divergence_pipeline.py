import torch as T

import src.metric.divergences as dvg
from src.scm.masked_scm import (
    DEFAULT_GATE_SHARPNESS,
    DEFAULT_MASK_MODE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MAX_ITERS,
    DEFAULT_TOL,
    DEFAULT_U_SIZE,
    DEFAULT_USE_DAG_UPDATES,
)
from src.scm.ncm.masked_feedforward_ncm import MaskedFF_NCM

from .masked_base_pipeline import MaskedBasePipeline


class MaskedDivergencePipeline(MaskedBasePipeline):
    patience = 50

    def __init__(
            self,
            generator,
            do_var_list,
            dat_sets,
            cg,
            dim,
            hyperparams=None,
            ncm_model=MaskedFF_NCM):
        if hyperparams is None:
            hyperparams = dict()

        v_size = {v: dim if v not in ('X', 'Y') else 1 for v in cg.v}
        ncm = ncm_model(
            v=list(cg.v),
            v_size=v_size,
            default_u_size=hyperparams.get('u-size', DEFAULT_U_SIZE),
            hyperparams=hyperparams,
            mask_mode=hyperparams.get('mask-mode', DEFAULT_MASK_MODE),
            mask_threshold=hyperparams.get('mask-threshold', DEFAULT_MASK_THRESHOLD),
            gate_sharpness=hyperparams.get('gate-sharpness', DEFAULT_GATE_SHARPNESS),
            max_iters=hyperparams.get('max-iters', DEFAULT_MAX_ITERS),
            tol=hyperparams.get('tol', DEFAULT_TOL),
        )
        super().__init__(
            generator,
            do_var_list,
            dat_sets,
            cg,
            dim,
            ncm,
            batch_size=hyperparams.get('data-bs', 1000),
            init_mask=hyperparams.get('init-mask', None),
            learn_mask=hyperparams.get('learn-mask', True),
            use_dag_updates=hyperparams.get('use-dag-updates', DEFAULT_USE_DAG_UPDATES),
        )

        self.ncm_batch_size = hyperparams.get('ncm-bs', 1000)
        self.lr = hyperparams.get('lr', 0.001)
        self.cycle_lambda = hyperparams.get('cycle-lambda', 0.0)
        self.cycle_penalty_type = hyperparams.get('cycle-penalty', 'notears')
        self.dagma_s = hyperparams.get('dagma-s', 1.0)
        self.ordered_v = cg.v
        self.logged = False
        self.automatic_optimization = False

    def configure_optimizers(self):
        optim = T.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            'optimizer': optim,
            'lr_scheduler': T.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, 50, 1, eta_min=1e-4)
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        ncm_batch = self._sample_ncm(n=self.ncm_batch_size)
        dat_mat = T.cat([batch[k] for k in self.ordered_v], axis=1)
        ncm_mat = T.cat([ncm_batch[k] for k in self.ordered_v], axis=1)

        opt.zero_grad()
        mmd_loss = dvg.MMD_loss(dat_mat.float(), ncm_mat.float(), gamma=1)
        cycle_loss = 0.0
        if self.cycle_lambda > 0:
            cycle_loss = self.cycle_lambda * self.dag_penalty(
                penalty_type=self.cycle_penalty_type,
                dagma_s=self.dagma_s)
        loss = mmd_loss + cycle_loss
        loss_val = loss.item()
        mmd_loss_val = mmd_loss.item()
        cycle_loss_val = cycle_loss.item() if T.is_tensor(cycle_loss) else cycle_loss
        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss_val, prog_bar=True)
        self.log('mmd_loss', mmd_loss_val, prog_bar=True)
        self.log('cycle_loss', cycle_loss_val, prog_bar=True)
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)
        return loss
