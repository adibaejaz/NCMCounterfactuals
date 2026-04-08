import torch as T

from src.scm.masked_scm import DEFAULT_USE_DAG_UPDATES
from src.scm.ncm.masked_feedforward_ncm import MaskedFF_NCM

from .base_pipeline import BasePipeline


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
            use_dag_updates=DEFAULT_USE_DAG_UPDATES):
        super().__init__(generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=batch_size)

        if init_mask is None:
            init_mask = T.ones((len(self.ncm.v), len(self.ncm.v)), dtype=T.float)

        init_mask = init_mask.float()
        if learn_mask:
            self.mask_parameter = T.nn.Parameter(init_mask)
        else:
            self.register_buffer("mask_parameter", init_mask)

        self.learn_mask = learn_mask
        self.use_dag_updates = use_dag_updates

    def get_mask(self):
        return self.mask_parameter

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
