import numpy as np
import torch as T

import src.metric.divergences as dvg
from src.ds.counterfactual import CTF
from src.metric.evaluation import all_metrics
from src.run.data_setup import _build_v_sizes
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
    DEFAULT_MASK_L1_LAMBDA,
    MaskedBasePipeline,
)

DEFAULT_MASK_BINARY_LAMBDA = 0.0
DEFAULT_MASK_NON_COLLIDER_LAMBDA = 0.1


class MaskedDivergencePipeline(MaskedBasePipeline):
    patience = 50

    def _mask_grad_norm(self, loss):
        if not self.learn_mask or not T.is_tensor(loss):
            return 0.0
        grad = T.autograd.grad(loss, self.mask_parameter, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            return 0.0
        return grad.norm().detach().item()

    def _theta_grad_norm(self, loss):
        if not T.is_tensor(loss):
            return 0.0
        theta_params = [param for param in self.theta_parameters() if param.requires_grad]
        if not theta_params:
            return 0.0
        grads = T.autograd.grad(loss, theta_params, retain_graph=True, allow_unused=True)
        squared_norm = 0.0
        for grad in grads:
            if grad is not None:
                squared_norm += grad.pow(2).sum().detach().item()
        return squared_norm ** 0.5

    def __init__(
            self,
            generator,
            do_var_list,
            dat_sets,
            cg,
            dim,
            hyperparams=None,
            ncm_model=MaskedFF_NCM,
            max_query=None):
        if hyperparams is None:
            hyperparams = dict()

        v_size = _build_v_sizes(cg, dim, hyperparams)
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
            fixed_zero_edges=hyperparams.get('mask-fixed-zero-edges', None),
            fixed_zero_mask=hyperparams.get('mask-fixed-zero-mask', None),
            fixed_one_edges=hyperparams.get('mask-fixed-one-edges', None),
            fixed_one_mask=hyperparams.get('mask-fixed-one-mask', None),
            coupled_edges=hyperparams.get('mask-coupled-edges', None),
            mask_init_mode=hyperparams.get('mask-init-mode', 'constant'),
            mask_init_value=hyperparams.get('mask-init-value', 0.5),
            mask_init_range=hyperparams.get('mask-init-range', (0.25, 0.75)),
            use_dag_updates=hyperparams.get('use-dag-updates', DEFAULT_USE_DAG_UPDATES),
            num_workers=hyperparams.get('num-workers', 0),
        )

        if isinstance(max_query, CTF):
            self.max_query = [max_query]
        else:
            self.max_query = max_query
        self.ncm_batch_size = hyperparams.get("ncm-bs", 1000)
        self.lr = hyperparams.get("lr", 0.001)
        self.theta_lr = hyperparams.get("theta-lr", self.lr)
        self.mask_lr = hyperparams.get("mask-lr", self.lr)
        self.max_query_iters = hyperparams.get("max-query-iters", 3000)
        self.mc_sample_size = hyperparams.get("mc-sample-size", 10000)
        self.min_lambda = hyperparams.get("min-lambda", 0.001)
        self.max_lambda = hyperparams.get("max-lambda", 1.0)
        self.selection_query_lambda = float(
            hyperparams.get("selection-query-lambda", self.min_lambda))
        if self.selection_query_lambda < 0:
            raise ValueError("selection-query-lambda must be nonnegative")
        self.query_update_target = hyperparams.get("query-update-target", "mask")
        if self.query_update_target not in {"mask", "theta", "all"}:
            raise ValueError("query-update-target must be one of: mask, theta, all")
        self.mask_fit_loss_weight = float(hyperparams.get("mask-fit-loss-weight", 1.0))
        if self.mask_fit_loss_weight < 0:
            raise ValueError("mask-fit-loss-weight must be nonnegative")
        self.alt_opt = hyperparams.get("alt-opt", False)
        self.theta_steps_per_mask = hyperparams.get("theta-steps-per-mask", 5)
        self.mask_steps_per_theta = hyperparams.get("mask-steps-per-theta", 1)
        self.log_grad_norms = hyperparams.get("log-grad-norms", False)
        self.dag_alm = hyperparams.get("dag-alm", False)
        self.cycle_lambda = hyperparams.get('cycle-lambda', DEFAULT_CYCLE_LAMBDA)
        self.cycle_penalty_type = hyperparams.get('cycle-penalty', DEFAULT_CYCLE_PENALTY)
        self.dagma_s = hyperparams.get('dagma-s', DEFAULT_DAGMA_S)
        self.alm_rho_mult = hyperparams.get('alm-rho-mult', 5.0)
        self.alm_rho_max = hyperparams.get('alm-rho-max', 1e4)
        self.alm_update_every = hyperparams.get('alm-update-every', 50)
        self.alm_improve_ratio = hyperparams.get('alm-improve-ratio', 0.9)
        self.register_buffer('alm_alpha', T.tensor(float(hyperparams.get('alm-alpha-init', 0.0))))
        self.register_buffer('alm_rho', T.tensor(float(hyperparams.get('alm-rho-init', 1.0))))
        self.register_buffer('alm_prev_h', T.tensor(float('inf')))
        self.mask_l1_lambda = hyperparams.get('mask-l1-lambda', DEFAULT_MASK_L1_LAMBDA)
        self.mask_binary_lambda = hyperparams.get('mask-binary-lambda', DEFAULT_MASK_BINARY_LAMBDA)
        self.mask_non_collider_triples = list(hyperparams.get('mask-non-collider-triples', []))
        self.mask_non_collider_lambda = hyperparams.get(
            'mask-non-collider-lambda', DEFAULT_MASK_NON_COLLIDER_LAMBDA)
        self._validate_non_collider_triples()
        self.theta_only_phase = hyperparams.get('theta-only-phase', False)
        self.final_query_reg = hyperparams.get('final-query-reg', False)
        self.ordered_v = cg.v
        self.logged = False
        self.automatic_optimization = False
        self._alm_h_sum = 0.0
        self._alm_h_count = 0

    def configure_optimizers(self):
        if self.theta_only_phase:
            return T.optim.AdamW(self.theta_parameters(), lr=self.theta_lr)
        if self.alt_opt:
            theta_optim = T.optim.AdamW(self.theta_parameters(), lr=self.theta_lr)
            mask_optim = T.optim.AdamW(self.mask_parameters(), lr=self.mask_lr)
            return [theta_optim, mask_optim]
        optim = T.optim.AdamW([
            {"params": self.theta_parameters(), "lr": self.theta_lr},
            {"params": self.mask_parameters(), "lr": self.mask_lr},
        ])
        return {
            "optimizer": optim,
            "lr_scheduler": T.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, 50, 1, eta_min=1e-4)
        }

    def _compute_fit_loss(self, batch):
        fit_loss = 0.0
        for i, do_set in enumerate(self.do_var_list):
            ncm_batch = self._sample_ncm(
                n=self.ncm_batch_size,
                do={k: expand_do(v, n=self.ncm_batch_size).to(self.device)
                    for (k, v) in do_set.items()}
            )
            dat_mat = T.cat([batch[i][k] for k in self.ordered_v], axis=1)
            ncm_mat = T.cat([ncm_batch[k] for k in self.ordered_v], axis=1)
            fit_loss = fit_loss + dvg.MMD_loss(dat_mat.float(), ncm_mat.float(), gamma=1) / len(self.do_var_list)
        return fit_loss

    def _query_reg_weight(self):
        if self.final_query_reg:
            return self.min_lambda
        reg_ratio = min(self.current_epoch, self.max_query_iters) / self.max_query_iters
        reg_up = np.log(self.max_lambda)
        reg_low = np.log(self.min_lambda)
        return np.exp(reg_up - reg_ratio * (reg_up - reg_low))

    def start_theta_only_phase(self, theta_lr=None, final_query_reg=True):
        if theta_lr is not None:
            self.theta_lr = theta_lr
        self.theta_only_phase = True
        self.final_query_reg = final_query_reg
        self.set_theta_requires_grad(True)
        self.set_mask_requires_grad(False)

    def _validate_non_collider_triples(self):
        variables = set(self.ncm.v)
        for triple in self.mask_non_collider_triples:
            if len(triple) != 3:
                raise ValueError("non-collider triple must contain exactly three variables")
            missing = set(triple).difference(variables)
            if missing:
                raise ValueError("unknown non-collider variable(s): {}".format(sorted(missing)))

    def mask_non_collider_penalty(self):
        mask = self.get_mask()
        penalty = mask.new_tensor(0.0)
        v2i = {v: i for i, v in enumerate(self.ncm.v)}
        for x, y, z in self.mask_non_collider_triples:
            penalty = penalty + mask[v2i[x], v2i[y]] * mask[v2i[z], v2i[y]]
        return penalty

    def _compute_structure_terms(self):
        cycle_loss = 0.0
        dag_h = 0.0
        mask_l1 = self.mask_l1_penalty()
        mask_l1_loss = self.mask_l1_lambda * mask_l1
        mask_binary = self.mask_binary_penalty()
        mask_binary_loss = self.mask_binary_lambda * mask_binary
        mask_non_collider = self.mask_non_collider_penalty()
        mask_non_collider_loss = self.mask_non_collider_lambda * mask_non_collider
        if self.dag_alm:
            dag_h = self.notears_dag_penalty()
            alpha = self.alm_alpha.to(device=dag_h.device, dtype=dag_h.dtype)
            rho = self.alm_rho.to(device=dag_h.device, dtype=dag_h.dtype)
            cycle_loss = alpha * dag_h + 0.5 * rho * dag_h.pow(2)
        elif self.cycle_lambda > 0:
            dag_h = self.dag_penalty(
                penalty_type=self.cycle_penalty_type,
                dagma_s=self.dagma_s)
            cycle_loss = self.cycle_lambda * dag_h
        return (
            cycle_loss, dag_h, mask_l1, mask_l1_loss, mask_binary,
            mask_binary_loss, mask_non_collider, mask_non_collider_loss)

    def _current_phase(self):
        if self.theta_only_phase:
            return "theta"
        if not self.alt_opt:
            return "joint"
        cycle_len = self.theta_steps_per_mask + self.mask_steps_per_theta
        if cycle_len <= 0:
            raise ValueError("theta/mask step counts must sum to a positive value")
        phase_step = self.global_step % cycle_len
        if phase_step < self.theta_steps_per_mask:
            return "theta"
        return "mask"

    def _get_q_loss(self):
        query_loss = 0
        for query in self.max_query:
            cur_loss = self.ncm.compute_ctf(
                query,
                n=self.mc_sample_size,
                mask=self.get_mask(),
                use_dag_updates=self.use_dag_updates)
            if T.isnan(cur_loss):
                return cur_loss
            query_loss += cur_loss
        return query_loss

    def _get_q_loss_with_theta_frozen(self):
        theta_params = list(self.theta_parameters())
        theta_requires_grad = [param.requires_grad for param in theta_params]
        for param in theta_params:
            param.requires_grad_(False)
        try:
            return self._get_q_loss()
        finally:
            for param, requires_grad in zip(theta_params, theta_requires_grad):
                param.requires_grad_(requires_grad)

    def _get_q_loss_with_mask_frozen(self):
        mask_params = list(self.mask_parameters())
        mask_requires_grad = [param.requires_grad for param in mask_params]
        for param in mask_params:
            param.requires_grad_(False)
        try:
            return self._get_q_loss()
        finally:
            for param, requires_grad in zip(mask_params, mask_requires_grad):
                param.requires_grad_(requires_grad)

    def _query_loss_enabled_for_phase(self, phase):
        if self.max_query is None:
            return False
        if self.theta_only_phase and not self.final_query_reg:
            return False
        return self.query_update_target == "all" or self.query_update_target == phase

    def _get_phase_q_loss(self, phase):
        if self.query_update_target == "mask" and phase == "joint":
            return self._get_q_loss_with_theta_frozen()
        if self.query_update_target == "theta" and phase == "joint":
            return self._get_q_loss_with_mask_frozen()
        return self._get_q_loss()


    def on_train_epoch_start(self):
        self._alm_h_sum = 0.0
        self._alm_h_count = 0
        self._selection_mmd_sum = 0.0
        self._selection_mmd_count = 0

    def on_train_epoch_end(self):
        if self.dag_alm:
            self._update_dag_alm()

    def _log_selection_loss(self):
        if self._selection_mmd_count <= 0:
            return
        with T.no_grad():
            selection_mmd_loss = self._selection_mmd_sum / self._selection_mmd_count
            (
                cycle_loss, _dag_h, _mask_l1, mask_l1_loss, _mask_binary,
                mask_binary_loss, _mask_non_collider,
                mask_non_collider_loss) = self._compute_structure_terms()
            selection_structure_loss = (
                cycle_loss + mask_l1_loss + mask_binary_loss + mask_non_collider_loss)
            selection_query_loss = self.mask_parameter.new_tensor(0.0)
            if self.max_query is not None and self.selection_query_lambda > 0:
                query_objective = self._get_q_loss()
                if not T.isnan(query_objective):
                    selection_query_loss = self.selection_query_lambda * query_objective
            selection_mmd_loss = selection_query_loss.new_tensor(selection_mmd_loss)
            selection_loss = (
                selection_mmd_loss + selection_structure_loss + selection_query_loss)

        self.log("selection_loss", selection_loss, prog_bar=True)
        self.log("selection_mmd_loss", selection_mmd_loss, prog_bar=False)
        self.log("selection_structure_loss", selection_structure_loss, prog_bar=False)
        self.log("selection_query_loss", selection_query_loss, prog_bar=False)
        self.log("selection_query_lambda", self.selection_query_lambda, prog_bar=False)

    def _update_dag_alm(self):
        if self._alm_h_count <= 0:
            return
        if (self.current_epoch + 1) % self.alm_update_every != 0:
            return
        mean_h = self._alm_h_sum / self._alm_h_count
        prev_h = float(self.alm_prev_h.item())
        if prev_h < float('inf') and mean_h > self.alm_improve_ratio * prev_h:
            new_rho = min(float(self.alm_rho.item()) * self.alm_rho_mult, self.alm_rho_max)
            self.alm_rho.fill_(new_rho)
        self.alm_alpha.add_(float(self.alm_rho.item()) * mean_h)
        self.alm_prev_h.fill_(mean_h)

    def training_step(self, batch, batch_idx):
        phase = self._current_phase()
        if self.theta_only_phase:
            active_opt = self.optimizers()
            self.set_theta_requires_grad(True)
            self.set_mask_requires_grad(False)
        elif self.alt_opt:
            theta_opt, mask_opt = self.optimizers()
            active_opt = theta_opt if phase == "theta" else mask_opt
            self.set_theta_requires_grad(phase == "theta")
            self.set_mask_requires_grad(phase == "mask")
        else:
            active_opt = self.optimizers()

        max_reg = self._query_reg_weight()
        (
            cycle_loss, dag_h, mask_l1, mask_l1_loss, mask_binary,
            mask_binary_loss, mask_non_collider,
            mask_non_collider_loss) = self._compute_structure_terms()

        q_loss = 0.0
        if self._query_loss_enabled_for_phase(phase):
            query_objective = self._get_phase_q_loss(phase)
            if not T.isnan(query_objective):
                q_loss = max_reg * query_objective

        active_opt.zero_grad()
        mmd_loss = self._compute_fit_loss(batch)
        self._selection_mmd_sum += mmd_loss.detach().item()
        self._selection_mmd_count += 1
        weighted_mmd_loss = mmd_loss
        if phase == "mask":
            weighted_mmd_loss = self.mask_fit_loss_weight * mmd_loss
        objective_loss = weighted_mmd_loss + q_loss
        structure_loss = (
            cycle_loss + mask_l1_loss + mask_binary_loss + mask_non_collider_loss)
        if phase == "mask":
            loss = objective_loss + structure_loss
        else:
            loss = objective_loss
        loss_val = loss.item()
        mmd_loss_val = mmd_loss.item()
        cycle_loss_val = cycle_loss.item() if T.is_tensor(cycle_loss) else cycle_loss
        dag_h_val = dag_h.item() if T.is_tensor(dag_h) else dag_h
        if self.dag_alm and T.is_tensor(dag_h):
            self._alm_h_sum += dag_h.detach().item()
            self._alm_h_count += 1
        mask_l1_val = mask_l1.item()
        mask_l1_loss_val = mask_l1_loss.item()
        mask_binary_val = mask_binary.item()
        mask_binary_loss_val = mask_binary_loss.item()
        mask_non_collider_val = mask_non_collider.item()
        mask_non_collider_loss_val = mask_non_collider_loss.item()
        q_loss_val = q_loss.item() if T.is_tensor(q_loss) else q_loss
        weighted_mmd_loss_val = weighted_mmd_loss.item() if T.is_tensor(weighted_mmd_loss) else weighted_mmd_loss
        objective_loss_val = objective_loss.item()
        structure_loss_val = structure_loss.item() if T.is_tensor(structure_loss) else structure_loss
        if self.log_grad_norms:
            mask_fit_grad_norm = self._mask_grad_norm(weighted_mmd_loss)
            mask_query_grad_norm = self._mask_grad_norm(q_loss)
            mask_dag_grad_norm = self._mask_grad_norm(cycle_loss)
            mask_total_grad_norm = self._mask_grad_norm(loss)
            theta_fit_grad_norm = self._theta_grad_norm(mmd_loss)
            theta_query_grad_norm = self._theta_grad_norm(q_loss)
            theta_total_grad_norm = self._theta_grad_norm(loss)
        self.manual_backward(loss)
        active_opt.step()

        if self.theta_only_phase:
            self.set_theta_requires_grad(True)
            self.set_mask_requires_grad(False)
        elif self.alt_opt:
            self.set_theta_requires_grad(True)
            self.set_mask_requires_grad(True)

        self.log("train_loss", loss_val, prog_bar=True)
        self.log("objective_loss", objective_loss_val, prog_bar=True)
        self.log("structure_loss", structure_loss_val, prog_bar=True)
        self.log("mmd_loss", mmd_loss_val, prog_bar=True)
        self.log("weighted_mmd_loss", weighted_mmd_loss_val, prog_bar=False)
        self.log("mask_fit_loss_weight", self.mask_fit_loss_weight if phase == "mask" else 1.0, prog_bar=False)
        self.log("dag_h", dag_h_val, prog_bar=True)
        self.log("cycle_loss", cycle_loss_val, prog_bar=True)
        if self.dag_alm:
            self.log("alm_alpha", self.alm_alpha.item(), prog_bar=False)
            self.log("alm_rho", self.alm_rho.item(), prog_bar=False)
        self.log("mask_l1", mask_l1_val, prog_bar=True)
        self.log("mask_l1_loss", mask_l1_loss_val, prog_bar=True)
        self.log("mask_binary", mask_binary_val, prog_bar=True)
        self.log("mask_binary_loss", mask_binary_loss_val, prog_bar=True)
        self.log("mask_non_collider", mask_non_collider_val, prog_bar=True)
        self.log("mask_non_collider_loss", mask_non_collider_loss_val, prog_bar=True)
        self.log("Q_loss", q_loss_val, prog_bar=True)
        if self.log_grad_norms:
            self.log("mask_fit_grad_norm", mask_fit_grad_norm, prog_bar=False)
            self.log("mask_query_grad_norm", mask_query_grad_norm, prog_bar=False)
            self.log("mask_dag_grad_norm", mask_dag_grad_norm, prog_bar=False)
            self.log("mask_total_grad_norm", mask_total_grad_norm, prog_bar=False)
            self.log("theta_fit_grad_norm", theta_fit_grad_norm, prog_bar=False)
            self.log("theta_query_grad_norm", theta_query_grad_norm, prog_bar=False)
            self.log("theta_total_grad_norm", theta_total_grad_norm, prog_bar=False)
        self.log("phase_is_mask", 1 if phase == "mask" else 0, prog_bar=True)
        if phase == "mask":
            self.log("train_loss_mask", loss_val, prog_bar=False)
        else:
            self.log("train_loss_theta", loss_val, prog_bar=False)
        if self.theta_only_phase:
            self.log("theta_lr", active_opt.param_groups[0]["lr"], prog_bar=True)
            self.log("mask_lr", 0.0, prog_bar=True)
        elif self.alt_opt:
            self.log("theta_lr", theta_opt.param_groups[0]["lr"], prog_bar=True)
            self.log("mask_lr", mask_opt.param_groups[0]["lr"], prog_bar=True)
        else:
            self.log("theta_lr", active_opt.param_groups[0]["lr"], prog_bar=True)
            self.log("mask_lr", active_opt.param_groups[1]["lr"], prog_bar=True)

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
