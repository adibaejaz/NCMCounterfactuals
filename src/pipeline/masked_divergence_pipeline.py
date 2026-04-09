import torch as T

import src.metric.divergences as dvg
from src.metric.evaluation import all_metrics
from src.scm.scm import expand_do
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

from .masked_base_pipeline import (
    DEFAULT_CYCLE_LAMBDA,
    DEFAULT_CYCLE_PENALTY,
    DEFAULT_DAGMA_S,
    MaskedBasePipeline,
)


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
            mask_init_mode=hyperparams.get('mask-init-mode', 'constant'),
            mask_init_value=hyperparams.get('mask-init-value', 0.5),
            mask_init_range=hyperparams.get('mask-init-range', (0.25, 0.75)),
            use_dag_updates=hyperparams.get('use-dag-updates', DEFAULT_USE_DAG_UPDATES),
        )

        self.ncm_batch_size = hyperparams.get('ncm-bs', 1000)
        self.lr = hyperparams.get('lr', 0.001)
        self.cycle_lambda = hyperparams.get('cycle-lambda', DEFAULT_CYCLE_LAMBDA)
        self.cycle_penalty_type = hyperparams.get('cycle-penalty', DEFAULT_CYCLE_PENALTY)
        self.dagma_s = hyperparams.get('dagma-s', DEFAULT_DAGMA_S)
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

        opt.zero_grad()
        mmd_loss = 0.0
        for i, do_set in enumerate(self.do_var_list):
            ncm_batch = self._sample_ncm(
                n=self.ncm_batch_size,
                do={k: expand_do(v, n=self.ncm_batch_size).to(self.device)
                    for (k, v) in do_set.items()}
            )
            dat_mat = T.cat([batch[i][k] for k in self.ordered_v], axis=1)
            ncm_mat = T.cat([ncm_batch[k] for k in self.ordered_v], axis=1)
            mmd_loss = mmd_loss + dvg.MMD_loss(dat_mat.float(), ncm_mat.float(), gamma=1) / len(self.do_var_list)
        cycle_loss = 0.0
        dag_h = 0.0
        if self.cycle_lambda > 0:
            dag_h = self.dag_penalty(
                penalty_type=self.cycle_penalty_type,
                dagma_s=self.dagma_s)
            cycle_loss = self.cycle_lambda * dag_h
        loss = mmd_loss + cycle_loss
        loss_val = loss.item()
        mmd_loss_val = mmd_loss.item()
        cycle_loss_val = cycle_loss.item() if T.is_tensor(cycle_loss) else cycle_loss
        dag_h_val = dag_h.item() if T.is_tensor(dag_h) else dag_h
        self.manual_backward(loss)
        opt.step()

        self.log('train_loss', loss_val, prog_bar=True)
        self.log('mmd_loss', mmd_loss_val, prog_bar=True)
        self.log('dag_h', dag_h_val, prog_bar=True)
        self.log('cycle_loss', cycle_loss_val, prog_bar=True)
        self.log('lr', opt.param_groups[0]['lr'], prog_bar=True)

        if (self.current_epoch + 1) % 10 == 0:
            if not self.logged:
                results = all_metrics(
                    self.generator,
                    self.ncm,
                    self.do_var_list,
                    self.dat_sets,
                    n=10000,
                    stored=self.stored_metrics,
                    ncm_kwargs=dict(
                        mask=self.get_mask(),
                        use_dag_updates=self.use_dag_updates,
                    ),
                )
                for k, v in results.items():
                    self.log(k, v)
                self.logged = True
        else:
            self.logged = False

        return loss


if __name__ == "__main__":
    dat = {"X": T.zeros((16, 1)), "Y": T.zeros((16, 1))}
    hyperparams = {
        "data-bs": 8,
        "ncm-bs": 8,
        "lr": 1e-3,
        "learn-mask": True,
        "mask-mode": "gate",
        "mask-init-mode": "constant",
        "mask-init-value": 0.5,
        "cycle-lambda": 0.1,
        "cycle-penalty": DEFAULT_CYCLE_PENALTY,
        "dagma-s": DEFAULT_DAGMA_S,
        "max-iters": 4,
        "tol": 1e-6,
    }

    class DummyGraph:
        def __init__(self, v):
            self.v = v

    pipeline = MaskedDivergencePipeline(
        generator=None,
        do_var_list=[{}],
        dat_sets=[dat],
        cg=DummyGraph(["X", "Y"]),
        dim=1,
        hyperparams=hyperparams,
    )

    class _DummyOpt:
        def __init__(self, params, lr):
            self.opt = T.optim.AdamW(params, lr=lr)
            self.param_groups = self.opt.param_groups

        def zero_grad(self):
            self.opt.zero_grad()

        def step(self):
            self.opt.step()

    dummy_opt = _DummyOpt(pipeline.parameters(), pipeline.lr)
    pipeline.optimizers = lambda: dummy_opt
    pipeline.manual_backward = lambda loss: loss.backward()

    batch = {"X": T.zeros((pipeline.batch_size, 1)), "Y": T.zeros((pipeline.batch_size, 1))}
    mask_before = pipeline.get_mask().detach().clone()
    loss = pipeline.training_step(batch, 0)
    mask_after = pipeline.get_mask().detach().clone()

    print("masked divergence one-step check")
    print("  loss:", round(loss.item(), 6))
    print("  dag_h:", round(pipeline.dag_penalty(
        penalty_type=pipeline.cycle_penalty_type,
        dagma_s=pipeline.dagma_s).item(), 6))
    print("  mask_before:", mask_before)
    print("  mask_after:", mask_after)
    print("  mask_delta_norm:", round((mask_after - mask_before).norm().item(), 6))
