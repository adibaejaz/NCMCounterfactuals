import torch as T
import torch.nn as nn

from src.scm.distribution.continuous_distribution import UniformDistribution
from src.scm.masked_scm import (
    DEFAULT_GATE_SHARPNESS,
    DEFAULT_H_LAYERS,
    DEFAULT_H_SIZE,
    DEFAULT_MASK_MODE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MAX_ITERS,
    DEFAULT_PERP_VALUE,
    DEFAULT_TOL,
    DEFAULT_U_SIZE,
    DEFAULT_V_SIZE,
    MaskedSCM,
    _run_demo_cases,
)
from src.scm.nn.mlp import MLP


class MaskedFF_NCM(MaskedSCM):
    """
    Feedforward masked NCM with one neural mechanism per endogenous variable.

    Each node mechanism receives:
    - all other endogenous variables as candidate inputs
    - a node-specific exogenous noise variable

    The active contribution of each endogenous input is controlled by the mask
    handling logic implemented in ``MaskedSCM``.
    """

    def __init__(
            self,
            v,
            v_size={},
            default_v_size=DEFAULT_V_SIZE,
            u_size={},
            default_u_size=DEFAULT_U_SIZE,
            f={},
            hyperparams=None,
            default_module=MLP,
            perp_value=DEFAULT_PERP_VALUE,
            mask_mode=DEFAULT_MASK_MODE,
            mask_threshold=DEFAULT_MASK_THRESHOLD,
            gate_sharpness=DEFAULT_GATE_SHARPNESS,
            max_iters=DEFAULT_MAX_ITERS,
            tol=DEFAULT_TOL):
        if hyperparams is None:
            hyperparams = dict()

        self.u_names = {k: "U_{}".format(k) for k in v} # one exogenous node for each variable
        self.v_size = {k: v_size.get(k, default_v_size) for k in v}
        self.u_size = {
            self.u_names[k]: u_size.get(self.u_names[k], default_u_size)
            for k in v
        }

        super().__init__(
            v=list(v),
            f=nn.ModuleDict({
                k: f[k] if k in f else default_module(
                    {k2: self.v_size[k2] for k2 in v if k2 != k},
                    {self.u_names[k]: self.u_size[self.u_names[k]]},
                    self.v_size[k],
                    h_size=hyperparams.get('h-size', DEFAULT_H_SIZE),
                    h_layers=hyperparams.get('h-layers', DEFAULT_H_LAYERS),
                    use_sigmoid=hyperparams.get('use-sigmoid', True),
                    use_layer_norm=hyperparams.get('layer-norm', True),
                )
                for k in v
            }),
            pu=UniformDistribution(list(self.u_names.values()), self.u_size),
            perp_value=perp_value,
            v_size=self.v_size,
            mask_mode=mask_mode,
            mask_threshold=mask_threshold,
            gate_sharpness=gate_sharpness,
            max_iters=max_iters,
            tol=tol)

    def convert_evaluation(self, samples):
        return {k: T.round(samples[k]) for k in samples}


def _build_demo_masked_ff_ncm(mask_mode: str) -> MaskedFF_NCM:
    return MaskedFF_NCM(
        v=["X", "Y", "Z"],
        v_size={"X": 1, "Y": 1, "Z": 1},
        perp_value=DEFAULT_PERP_VALUE,
        mask_mode=mask_mode,
        mask_threshold=DEFAULT_MASK_THRESHOLD,
        gate_sharpness=DEFAULT_GATE_SHARPNESS,
        max_iters=6,
        tol=1e-6,
    )


if __name__ == "__main__":
    _run_demo_cases(_build_demo_masked_ff_ncm)
