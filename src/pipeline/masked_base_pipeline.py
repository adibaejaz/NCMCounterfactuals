import torch as T
from torch.special import logit

from src.scm.masked_scm import DEFAULT_USE_DAG_UPDATES
from src.scm.ncm.masked_feedforward_ncm import MaskedFF_NCM

from .base_pipeline import BasePipeline

DEFAULT_CYCLE_LAMBDA = 0.0
DEFAULT_CYCLE_PENALTY = "dagma"
DEFAULT_DAGMA_S = 1.0
DEFAULT_MASK_L1_LAMBDA = 1.0


class MaskedBasePipeline(BasePipeline):
    def __init__(
            self,
            generator,
            do_var_list,
            dat_sets,
            cg,
            dim,
            ncm,
            batch_size=256,
            init_mask=None,
            learn_mask=True,
            fixed_zero_edges=None,
            fixed_zero_mask=None,
            fixed_one_edges=None,
            fixed_one_mask=None,
            coupled_edges=None,
            mask_init_mode="constant",
            mask_init_value=0.5,
            mask_init_range=(0.25, 0.75),
            use_dag_updates=DEFAULT_USE_DAG_UPDATES):
        super().__init__(generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=batch_size)

        self.fixed_zero_mask, self.fixed_one_mask = self._build_fixed_masks(
            fixed_zero_edges=fixed_zero_edges,
            fixed_zero_mask=fixed_zero_mask,
            fixed_one_edges=fixed_one_edges,
            fixed_one_mask=fixed_one_mask,
        )
        self.coupled_edge_indices = self._build_coupled_edge_indices(coupled_edges)
        self._validate_coupled_edges()

        if init_mask is None:
            init_mask = self._init_mask(
                mode=mask_init_mode,
                value=mask_init_value,
                value_range=mask_init_range)

        init_mask = self._apply_fixed_masks(init_mask.float())
        if learn_mask:
            eps = 1e-6
            init_logits = logit(init_mask.clamp(min=eps, max=1 - eps))
            self.mask_parameter = T.nn.Parameter(init_logits)
            if self.coupled_edge_indices:
                init_coupled = T.tensor(
                    [init_mask[i, j].item() for (i, j) in self.coupled_edge_indices],
                    dtype=T.float)
                init_coupled_logits = logit(init_coupled.clamp(min=eps, max=1 - eps))
                self.coupled_mask_parameter = T.nn.Parameter(init_coupled_logits)
            else:
                self.register_buffer("coupled_mask_parameter", T.empty(0))
        else:
            self.register_buffer("mask_parameter", init_mask)
            self.register_buffer("coupled_mask_parameter", T.empty(0))

        self.learn_mask = learn_mask
        self.use_dag_updates = use_dag_updates

    def mask_parameters(self):
        if not self.learn_mask:
            return []
        params = [self.mask_parameter]
        if self.coupled_edge_indices:
            params.append(self.coupled_mask_parameter)
        return params

    def theta_parameters(self):
        mask_param_ids = {id(param) for param in self.mask_parameters()}
        return [
            param
            for param in self.parameters()
            if id(param) not in mask_param_ids
        ]

    def set_mask_requires_grad(self, flag: bool):
        for param in self.mask_parameters():
            param.requires_grad_(flag)

    def set_theta_requires_grad(self, flag: bool):
        for param in self.theta_parameters():
            param.requires_grad_(flag)

    def _build_fixed_masks(
            self,
            fixed_zero_edges=None,
            fixed_zero_mask=None,
            fixed_one_edges=None,
            fixed_one_mask=None):
        n = len(self.ncm.v)
        zero_mask = T.ones((n, n), dtype=T.float)
        one_mask = T.zeros((n, n), dtype=T.float)

        eye = T.eye(n, dtype=T.float)
        zero_mask = zero_mask * (1 - eye)

        if fixed_zero_mask is not None:
            fixed_zero_mask = fixed_zero_mask.float()
            if fixed_zero_mask.shape != (n, n):
                raise ValueError(
                    "fixed_zero_mask must have shape ({}, {}), got {}".format(
                        n, n, tuple(fixed_zero_mask.shape)))
            zero_mask = zero_mask * fixed_zero_mask

        if fixed_zero_edges is not None:
            v2i = {v: i for i, v in enumerate(self.ncm.v)}
            for src, dst in fixed_zero_edges:
                if src not in v2i or dst not in v2i:
                    raise ValueError("unknown fixed-zero edge {} -> {}".format(src, dst))
                zero_mask[v2i[src], v2i[dst]] = 0.0

        if fixed_one_mask is not None:
            fixed_one_mask = fixed_one_mask.float()
            if fixed_one_mask.shape != (n, n):
                raise ValueError(
                    "fixed_one_mask must have shape ({}, {}), got {}".format(
                        n, n, tuple(fixed_one_mask.shape)))
            one_mask = T.maximum(one_mask, fixed_one_mask)

        if fixed_one_edges is not None:
            v2i = {v: i for i, v in enumerate(self.ncm.v)}
            for src, dst in fixed_one_edges:
                if src not in v2i or dst not in v2i:
                    raise ValueError("unknown fixed-one edge {} -> {}".format(src, dst))
                one_mask[v2i[src], v2i[dst]] = 1.0

        one_mask = (one_mask > 0).float()
        if T.any(one_mask * eye):
            raise ValueError("fixed-one mask cannot include self-edges")
        if T.any(one_mask * (1 - zero_mask)):
            raise ValueError("an edge cannot be fixed to both 0 and 1")

        self.register_buffer("fixed_zero_mask", zero_mask)
        self.register_buffer("fixed_one_mask", one_mask)
        return self.fixed_zero_mask, self.fixed_one_mask

    def _build_coupled_edge_indices(self, coupled_edges=None):
        if coupled_edges is None:
            coupled_edges = []
        v2i = {v: i for i, v in enumerate(self.ncm.v)}
        indices = []
        seen = set()
        for src, dst in coupled_edges:
            if src not in v2i or dst not in v2i:
                raise ValueError("unknown coupled edge {} -- {}".format(src, dst))
            if src == dst:
                raise ValueError("coupled edge cannot be a self-edge: {}".format(src))
            canon_src, canon_dst = tuple(sorted((src, dst)))
            key = (canon_src, canon_dst)
            if key in seen:
                continue
            seen.add(key)
            indices.append((v2i[canon_src], v2i[canon_dst]))
        return indices

    def _validate_coupled_edges(self):
        for i, j in self.coupled_edge_indices:
            if self.fixed_zero_mask[i, j].item() == 0 or self.fixed_zero_mask[j, i].item() == 0:
                raise ValueError(
                    "coupled edge {} -- {} conflicts with fixed-zero constraints".format(
                        self.ncm.v[i], self.ncm.v[j]))
            if self.fixed_one_mask[i, j].item() != 0 or self.fixed_one_mask[j, i].item() != 0:
                raise ValueError(
                    "coupled edge {} -- {} conflicts with fixed-one constraints".format(
                        self.ncm.v[i], self.ncm.v[j]))

    def _init_mask(self, mode="constant", value=0.5, value_range=(0.25, 0.75)):
        n = len(self.ncm.v)
        eye = T.eye(n, dtype=T.float)

        if mode == "constant":
            mask = T.ones((n, n), dtype=T.float) * value
        elif mode == "uniform":
            low, high = value_range
            mask = low + (high - low) * T.rand((n, n), dtype=T.float)
        elif mode == "zeros":
            mask = T.zeros((n, n), dtype=T.float)
        elif mode == "ones":
            mask = T.ones((n, n), dtype=T.float)
        else:
            raise ValueError("unknown mask init mode: {}".format(mode))

        return mask * (1 - eye)

    def _apply_fixed_masks(self, mask):
        zero_mask = self.fixed_zero_mask.to(device=mask.device, dtype=mask.dtype)
        one_mask = self.fixed_one_mask.to(device=mask.device, dtype=mask.dtype)
        return mask * zero_mask * (1 - one_mask) + one_mask

    def _apply_coupled_edges(self, mask):
        if not self.coupled_edge_indices:
            return mask
        mask = mask.clone()
        if self.learn_mask:
            coupled_values = T.sigmoid(self.coupled_mask_parameter).to(
                device=mask.device, dtype=mask.dtype)
        else:
            coupled_values = T.stack([
                mask[i, j] for (i, j) in self.coupled_edge_indices
            ])
        for k, (i, j) in enumerate(self.coupled_edge_indices):
            value = coupled_values[k]
            mask[i, j] = value
            mask[j, i] = 1 - value
        return mask

    def get_mask(self):
        if self.learn_mask:
            mask = T.sigmoid(self.mask_parameter)
        else:
            mask = self.mask_parameter
        mask = self._apply_coupled_edges(mask)
        return self._apply_fixed_masks(mask)

    def get_edge_scores(self):
        return self.get_mask()

    def mask_l1_penalty(self):
        return self.get_mask().abs().sum()

    def mask_binary_penalty(self):
        mask = self.get_mask()
        return (mask * (1 - mask)).sum()

    def get_dagma_edge_scores(self, eps=1e-8):
        edge_scores = self.get_edge_scores()
        scale = edge_scores.abs().sum()
        if scale.item() <= eps:
            return edge_scores
        return edge_scores / scale.clamp_min(eps)

    def notears_dag_penalty(self):
        edge_scores = self.get_edge_scores()
        return T.trace(T.matrix_exp(edge_scores)) - edge_scores.shape[0]

    def dagma_dag_penalty(self, s=1.0):
        edge_scores = self.get_dagma_edge_scores()
        d = edge_scores.shape[0]
        identity = T.eye(d, device=edge_scores.device, dtype=edge_scores.dtype)
        s_tensor = T.tensor(float(s), device=edge_scores.device, dtype=edge_scores.dtype)
        matrix = s_tensor * identity - edge_scores
        sign, logabsdet = T.linalg.slogdet(matrix)
        if sign <= 0:
            return T.tensor(float("inf"), device=edge_scores.device, dtype=edge_scores.dtype)
        return -logabsdet + d * T.log(s_tensor)

    def dag_penalty(self, penalty_type="notears", dagma_s=1.0):
        if penalty_type == "notears":
            return self.notears_dag_penalty()
        if penalty_type == "dagma":
            return self.dagma_dag_penalty(s=dagma_s)
        raise ValueError("unknown DAG penalty type: {}".format(penalty_type))

    def _sample_ncm(self, n=None, u=None, do={}, evaluating=False):
        return self.ncm(
            n=n,
            u=u,
            do=do,
            evaluating=evaluating,
            mask=self.get_mask(),
            use_dag_updates=self.use_dag_updates)

    def forward(self, n=None, u=None, do={}, evaluating=False):
        return self._sample_ncm(n=n, u=u, do=do, evaluating=evaluating)


if __name__ == "__main__":
    ncm = MaskedFF_NCM(
        v=["X", "Y"],
        v_size={"X": 1, "Y": 1},
        mask_mode="gate",
        max_iters=4,
        tol=1e-6,
    )

    init_mask = T.tensor([
        [0.0, 0.5],
        [0.5, 0.0],
    ], dtype=T.float)

    dummy_data = [{"X": T.zeros((8, 1)), "Y": T.zeros((8, 1))}]

    pipeline = MaskedBasePipeline(
        generator=None,
        do_var_list=[{}],
        dat_sets=dummy_data,
        cg=None,
        dim=1,
        ncm=ncm,
        init_mask=init_mask,
        learn_mask=True,
        use_dag_updates=False,
    )

    u = pipeline.ncm.pu.sample(32)
    mask_before = pipeline.get_mask().detach().clone()
    out_before = pipeline(u=u)
    y_mean_before = out_before["Y"].float().mean().item()

    loss = -out_before["Y"].float().mean()
    loss.backward()
    grad_norm = pipeline.mask_parameter.grad.norm().item()

    opt = T.optim.Adam([pipeline.mask_parameter], lr=0.1)
    opt.step()
    opt.zero_grad()

    mask_after = pipeline.get_mask().detach().clone()
    out_after = pipeline(u=u)
    y_mean_after = out_after["Y"].float().mean().item()

    print("learnable mask check")
    print("  y_mean_before:", round(y_mean_before, 6))
    print("  y_mean_after:", round(y_mean_after, 6))
    print("  grad_norm:", round(grad_norm, 6))
    print("  mask_before:", mask_before)
    print("  mask_after:", mask_after)
    print("  mask_delta_norm:", round((mask_after - mask_before).norm().item(), 6))

    fixed_pipeline = MaskedBasePipeline(
        generator=None,
        do_var_list=[{}],
        dat_sets=dummy_data,
        cg=None,
        dim=1,
        ncm=MaskedFF_NCM(v=["X", "Y"], v_size={"X": 1, "Y": 1}),
        init_mask=init_mask,
        learn_mask=False,
    )

    print("fixed mask check")
    print("  named_parameters:", [name for (name, _) in fixed_pipeline.named_parameters()])
    print("  named_buffers:", [name for (name, _) in fixed_pipeline.named_buffers() if "mask_parameter" in name])
