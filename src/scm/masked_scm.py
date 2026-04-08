from __future__ import annotations

from collections import deque
from typing import Dict, Literal, Mapping, Optional, Protocol, Sequence, Union

import torch as T
import torch.nn as nn

from .distribution.distribution import Distribution


TensorDict = Dict[str, T.Tensor]
MaskValue = Union[float, int, T.Tensor]
MaskTensor = T.Tensor


class StructuralFn(Protocol):
    def __call__(self, v: Mapping[str, T.Tensor], u: Mapping[str, T.Tensor]) -> T.Tensor:
        ...


class MaskedSCM(nn.Module):
    """
    Structural causal model with formal mask variables.

    The model is indexed by a real-valued ``n x n`` mask tensor, where
    ``mask[i, j]`` controls how variable ``self.v[i]`` is exposed to the
    structural mechanism for ``self.v[j]``. The intended use is a soft mask in
    the unit interval, although this class does not enforce bounds.

    Structural mechanisms keep the same callable type as in a standard SCM:
    ``f_k(v, u)``. The difference is that each node is evaluated against a
    node-local masked view of ``v``. The rule that maps a mask entry and a
    value to the visible input is intentionally left abstract here so different
    masking schemes can be plugged in later.

    Likewise, this class does not fix an evaluation protocol. Different
    protocols may be needed for recursive, cyclic, synchronous, or fixed-point
    execution. This base implementation provides the shared bookkeeping and the
    node-local masking logic that those protocols should use.
    """

    def __init__(
            self,
            v: Sequence[str],
            f: Union[Mapping[str, StructuralFn], nn.ModuleDict],
            pu: Distribution,
            perp_value: MaskValue,
            v_size: Optional[Mapping[str, int]] = None,
            mask_mode: Literal["threshold", "multiply", "gate"] = "threshold",
            mask_threshold: float = 0.5,
            gate_sharpness: float = 10.0,
            max_iters: int = 1,
            tol: Optional[float] = None):
        """
        Args:
            v: Endogenous variable names.
            f: Structural mechanisms keyed by endogenous variable name.
            pu: Exogenous distribution.
            v_size: Optional endogenous variable dimensions. Defaults to 1 for
                any variable not listed.
            perp_value: Sentinel value reserved for masked-out inputs.
            mask_mode: Masking scheme applied to each directed input.
            mask_threshold: Threshold used by ``threshold`` and ``gate``.
            gate_sharpness: Sigmoid sharpness used by ``gate``.
            max_iters: Maximum number of synchronous update iterations.
            tol: Optional convergence tolerance. If provided, synchronous
                updates stop early once the maximum state change is at most
                ``tol``.

        Notes:
            The intended mask parameterization is an ``n x n`` tensor with
            values in ``[0, 1]``. This class only assumes that the same tensor
            shape is used consistently; it does not clip, rescale, or binarize
            the mask.
        """
        super().__init__()
        self.v = list(v)
        self.u = list(pu)
        self.f = f
        self.pu = pu
        if v_size is None:
            v_size = {}
        self.v_size = {k: v_size.get(k, 1) for k in self.v}
        self.perp_value = perp_value
        self.mask_mode = mask_mode
        self.mask_threshold = mask_threshold
        self.gate_sharpness = gate_sharpness
        self.max_iters = max_iters
        self.tol = tol
        self.v2i = {name: i for i, name in enumerate(self.v)}
        self.device_param = nn.Parameter(T.empty(0))
        self.last_converged = False
        self.last_iterations = 0
        self.last_delta = None

    def _normalize_mask(self, mask: MaskTensor) -> MaskTensor:
        """
        Put a mask tensor into the canonical internal form.

        This method does not modify mask values semantically. It only moves the
        tensor to the module device and checks that it is a rank-2 tensor of
        shape ``(len(self.v), len(self.v))``.
        """
        mask = mask.to(self.device_param)
        if mask.ndim != 2:
            raise ValueError("mask must be a rank-2 tensor")
        if mask.shape != (len(self.v), len(self.v)):
            raise ValueError(
                "mask must have shape ({}, {}), got {}".format(
                    len(self.v), len(self.v), tuple(mask.shape)))
        return mask

    def _normalize_do(self, do: Optional[TensorDict]) -> TensorDict:
        if do is None:
            do = {}
        if set(do.keys()).difference(self.v):
            raise ValueError("do contains variables outside the endogenous set")
        return {k: do[k].to(self.device_param) for k in do}

    def _normalize_select(self, select: Optional[Sequence[str]]) -> Sequence[str]:
        if select is None:
            return self.v
        return select

    def _normalize_u(self, n: Optional[int], u: Optional[TensorDict]) -> TensorDict:
        if (n is None) == (u is None):
            raise ValueError("exactly one of n or u must be provided")
        if u is not None:
            return u
        return self.pu.sample(n)

    def _perp_like(self, value: T.Tensor) -> T.Tensor:
        if T.is_tensor(self.perp_value):
            perp_value = self.perp_value.to(device=value.device, dtype=value.dtype)
        else:
            perp_value = T.tensor(self.perp_value, device=value.device, dtype=value.dtype)
        return T.ones_like(value) * perp_value

    def _threshold_mask(self, mask_value: T.Tensor, value: T.Tensor) -> T.Tensor:
        if mask_value.detach().item() > self.mask_threshold:
            return value
        return self._perp_like(value)

    def _multiply_mask(self, mask_value: T.Tensor, value: T.Tensor) -> T.Tensor:
        return mask_value.to(device=value.device, dtype=value.dtype) * value

    def _gate_mask(self, mask_value: T.Tensor, value: T.Tensor) -> T.Tensor:
        gate = T.sigmoid(
            self.gate_sharpness * (
                mask_value.to(device=value.device, dtype=value.dtype) - self.mask_threshold))
        perp_value = self._perp_like(value)
        return gate * value + (1 - gate) * perp_value

    def _edge_is_active(self, mask_value: T.Tensor) -> bool:
        """
        Decide whether a mask entry induces a directed edge in the associated
        graph.

        The current convention is:
        - ``threshold``: active iff ``mask_value > mask_threshold``
        - ``multiply``: active iff ``mask_value`` is nonzero
        - ``gate``: active iff ``mask_value > mask_threshold``
        """
        value = mask_value.detach().item()
        if self.mask_mode == "threshold":
            return bool(value > self.mask_threshold)
        if self.mask_mode == "multiply":
            return bool(value != 0)
        if self.mask_mode == "gate":
            return bool(value > self.mask_threshold)
        raise ValueError("unknown mask mode: {}".format(self.mask_mode))

    def _mask_value(self, mask_value: T.Tensor, value: T.Tensor) -> T.Tensor:
        """
        Apply the masking rule for a single directed input.

        Supported schemes are:
        - ``threshold``: return ``value`` when ``mask_value > mask_threshold``,
          otherwise return ``perp_value``
        - ``multiply``: return ``mask_value * value``
        - ``gate``: return a sigmoid interpolation between ``value`` and
          ``perp_value``, centered at ``mask_threshold``
        """
        if self.mask_mode == "threshold":
            return self._threshold_mask(mask_value, value)
        if self.mask_mode == "multiply":
            return self._multiply_mask(mask_value, value)
        if self.mask_mode == "gate":
            return self._gate_mask(mask_value, value)
        raise ValueError("unknown mask mode: {}".format(self.mask_mode))

    def _masked_inputs(self, k: str, v: Mapping[str, T.Tensor], mask: MaskTensor) -> TensorDict:
        """
        Build the masked node-local view of ``v`` used when evaluating node ``k``.

        For each non-self variable ``src`` that is present in ``v``, the value
        passed to ``f[k]`` is computed from ``mask[self.v2i[src], self.v2i[k]]``
        and the current value ``v[src]`` via ``_mask_value``.
        """
        j = self.v2i[k]
        v_masked = dict(v)
        for src in self.v:
            if src == k or src not in v:
                continue
            i = self.v2i[src]
            v_masked[src] = self._mask_value(mask[i, j], v[src])
        return v_masked

    def induced_edges(self, mask: MaskTensor):
        """
        Return the directed edges induced by ``mask`` under the active masking
        rule.
        """
        mask = self._normalize_mask(mask)
        edges = []
        for src in self.v:
            i = self.v2i[src]
            for dst in self.v:
                if src == dst:
                    continue
                j = self.v2i[dst]
                if self._edge_is_active(mask[i, j]):
                    edges.append((src, dst))
        return edges

    def is_acyclic(self, mask: MaskTensor) -> bool:
        """
        Check whether the directed graph induced by ``mask`` is acyclic.
        """
        parents = {k: set() for k in self.v}
        children = {k: set() for k in self.v}

        for src, dst in self.induced_edges(mask):
            parents[dst].add(src)
            children[src].add(dst)

        q = deque([k for k in self.v if len(parents[k]) == 0])
        visited = 0

        while q:
            cur = q.popleft()
            visited += 1
            for child in children[cur]:
                parents[child].remove(cur)
                if len(parents[child]) == 0:
                    q.append(child)

        return visited == len(self.v)

    def _evaluation_order(self, mask: MaskTensor):
        """
        Return a deterministic evaluation order induced by ``mask``.

        If the induced graph is acyclic, this is an ordinary topological order
        with ties broken by the original variable order ``self.v``. If the
        induced graph is cyclic, the method continues producing an order by
        selecting the earliest remaining variable in ``self.v`` whenever no
        parent-free node exists.
        """
        parents = {k: set() for k in self.v}
        children = {k: set() for k in self.v}

        for src, dst in self.induced_edges(mask):
            parents[dst].add(src)
            children[src].add(dst)

        order = []
        remaining = set(self.v)

        while remaining:
            frontier = [k for k in self.v if k in remaining and len(parents[k]) == 0]
            if frontier:
                cur = frontier[0]
            else:
                cur = next(k for k in self.v if k in remaining)

            order.append(cur)
            remaining.remove(cur)
            for child in children[cur]:
                if cur in parents[child]:
                    parents[child].remove(cur)

        return order

    def _compute_node(
            self,
            k: str,
            v: Mapping[str, T.Tensor],
            u: Mapping[str, T.Tensor],
            mask: MaskTensor) -> T.Tensor:
        v_masked = self._masked_inputs(k, v, mask)
        return self.f[k](v_masked, u)

    def _init_state(self, u: TensorDict, do: TensorDict) -> TensorDict:
        u_key = next(iter(u))
        n = len(u[u_key])
        state = {
            k: self._perp_like(T.ones((n, self.v_size[k]), device=self.device_param.device))
            for k in self.v
        }
        for k in do:
            state[k] = do[k]
        return state

    def _sync_step(
            self,
            v: TensorDict,
            u: TensorDict,
            do: TensorDict,
            mask: MaskTensor) -> TensorDict:
        v_next = {}
        for k in self.v:
            v_next[k] = do[k] if k in do else self._compute_node(k, v, u, mask)
        return v_next

    def _state_delta(self, v: TensorDict, v_next: TensorDict) -> float:
        delta = 0.0
        for k in self.v:
            new_delta = T.max(T.abs(v_next[k] - v[k])).detach().item()
            if new_delta > delta:
                delta = new_delta
        return delta

    def _sample_dag(
            self,
            u: TensorDict,
            do: TensorDict,
            select: Sequence[str],
            mask: MaskTensor):
        """
        Execute a single recursive sweep in the fixed variable order ``self.v``.

        This matches the update style of the standard SCM sampler, except that
        each node reads a masked view of the currently available endogenous
        state. No acyclicity check is performed; if the mask induces a cycle,
        the result still follows this fixed-order one-pass evaluation rule.
        """
        v = {}
        remaining = set(select)
        self.last_converged = False
        self.last_iterations = 1
        self.last_delta = None

        for k in self._evaluation_order(mask):
            v[k] = do[k] if k in do else self._compute_node(k, v, u, mask)
            remaining.discard(k)
            if not remaining:
                break

        return {k: v[k] for k in select}

    def sample(
            self,
            n: Optional[int] = None,
            u: Optional[TensorDict] = None,
            do: Optional[TensorDict] = None,
            select: Optional[Sequence[str]] = None,
            mask: Optional[MaskTensor] = None,
            use_dag_updates: bool = False):
        """
        Sample from the masked SCM using either synchronous or one-pass DAG
        updates.

        Exactly one of ``n`` or ``u`` must be provided. If ``u`` is not given,
        exogenous variables are sampled from ``self.pu``.

        If ``use_dag_updates`` is ``False``, the endogenous state is initialized
        to ``perp_value`` for every variable, with interventions in ``do``
        applied immediately and then held fixed throughout all synchronous
        updates. Each iteration computes a full next state from the previous
        full state. If ``tol`` is ``None``, exactly ``max_iters`` iterations
        are run. If ``tol`` is provided, iteration stops early once the maximum
        absolute state change across endogenous variables is at most ``tol``.

        If ``use_dag_updates`` is ``True``, the model is evaluated in a single
        sweep following ``self.v``, analogous to a standard recursive SCM. This
        mode is allowed even when the induced mask graph is cyclic; in that case
        the result is defined by the fixed order rather than by graph
        acyclicity.
        """
        if mask is None:
            raise ValueError("mask must be provided")

        mask = self._normalize_mask(mask)
        do = self._normalize_do(do)
        select = self._normalize_select(select)
        u = self._normalize_u(n, u)
        if use_dag_updates:
            return self._sample_dag(u=u, do=do, select=select, mask=mask)
        return self._sample_impl(u=u, do=do, select=select, mask=mask)

    def _sample_impl(
            self,
            u: TensorDict,
            do: TensorDict,
            select: Sequence[str],
            mask: MaskTensor):
        """
        Execute synchronous masked updates from a ``perp`` initialization.

        The inputs are already normalized:
        - ``u`` is present
        - ``do`` contains only endogenous variables and is on the module device
        - ``select`` is not ``None``
        - ``mask`` is a rank-2 ``n x n`` tensor on the module device

        Returns:
            A dictionary containing only the variables listed in ``select`` from
            the final synchronous state.
        """
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")

        v = self._init_state(u, do)
        self.last_converged = False
        self.last_iterations = 0
        self.last_delta = None

        for iteration in range(self.max_iters):
            v_next = self._sync_step(v, u, do, mask)
            delta = self._state_delta(v, v_next)

            v = v_next
            self.last_iterations = iteration + 1
            self.last_delta = delta

            if self.tol is not None and delta <= self.tol:
                self.last_converged = True
                break

        return {k: v[k] for k in select}

    def convert_evaluation(self, samples: TensorDict) -> TensorDict:
        return samples

    def forward(
            self,
            n: Optional[int] = None,
            u: Optional[TensorDict] = None,
            do: Optional[TensorDict] = None,
            select: Optional[Sequence[str]] = None,
            evaluating: bool = False,
            mask: Optional[MaskTensor] = None,
            use_dag_updates: bool = False):
        if evaluating:
            with T.no_grad():
                result = self.sample(
                    n=n, u=u, do=do, select=select, mask=mask, use_dag_updates=use_dag_updates)
                result = self.convert_evaluation(result)
                return {k: result[k].cpu() for k in result}

        return self.sample(
            n=n, u=u, do=do, select=select, mask=mask, use_dag_updates=use_dag_updates)
