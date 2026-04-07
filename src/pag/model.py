import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, activation=nn.GELU):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class LearnableDAGChainAttentionMask(nn.Module):
    def __init__(
        self,
        seq_len,
        graph="chain",
        mask_structure="learnable",
        dagma_s=2.0,
        hard_mask_value=1e4,
        init_edge_logit=0,
        init_edge_logit_std=1.0,
        adjacency_normalization="none",
        gate_floor=0.05,
        gate_renorm_eps=1e-6,
    ):
        super().__init__()
        if graph != "chain":
            raise ValueError("Only graph='chain' is supported right now.")
        if seq_len != 6:
            raise ValueError("The chain mask only supports three PAG variables X, Y, Z.")
        if mask_structure not in {"learnable", "fixed-chain"}:
            raise ValueError("mask_structure must be 'learnable' or 'fixed-chain'.")
        if adjacency_normalization not in {"none", "sum"}:
            raise ValueError("adjacency_normalization must be 'none' or 'sum'.")
        if dagma_s <= 0:
            raise ValueError("dagma_s must be positive.")
        if gate_floor < 0 or gate_floor >= 1:
            raise ValueError("gate_floor must lie in [0, 1).")
        if gate_renorm_eps <= 0:
            raise ValueError("gate_renorm_eps must be positive.")

        self.mask_structure = mask_structure
        self.seq_len = seq_len
        self.dagma_s = dagma_s
        self.hard_mask_value = hard_mask_value
        self.adjacency_normalization = adjacency_normalization
        self.gate_floor = gate_floor
        self.gate_renorm_eps = gate_renorm_eps

        # Token order is fixed as [u_x, u_y, u_z, X, Y, Z].
        # Hard support:
        # u_x <- {u_x, Y}
        # u_y <- {u_y, X, Z}
        # u_z <- {u_z, Y}
        # X, Y, Z are self-only.
        allow = T.zeros(seq_len, seq_len, dtype=T.bool)
        allow[0, 0] = True
        allow[0, 4] = True
        allow[1, 1] = True
        allow[1, 3] = True
        allow[1, 5] = True
        allow[2, 2] = True
        allow[2, 4] = True
        allow[3, 3] = True
        allow[4, 4] = True
        allow[5, 5] = True
        self.register_buffer('allow_mask', allow)
        hard_mask = T.full((seq_len, seq_len), -hard_mask_value)
        hard_mask = T.where(allow, T.zeros_like(hard_mask), hard_mask)
        self.register_buffer('hard_mask', hard_mask)

        # Source -> target adjacency over observed variables [X, Y, Z] for chain graph.
        if self.mask_structure == "fixed-chain":
            fixed_adjacency = T.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ])
            self.register_buffer('fixed_adjacency', fixed_adjacency)
            self.register_parameter('edge_logits', None)
        else:
            edge_logits = T.full((4,), init_edge_logit)
            if init_edge_logit_std > 0:
                edge_logits = edge_logits + init_edge_logit_std * T.randn(4)
            self.edge_logits = nn.Parameter(edge_logits)
        self.register_buffer('frozen_adjacency', T.zeros(3, 3))
        self.register_buffer('has_frozen_adjacency', T.tensor(False, dtype=T.bool))

    def _uses_hard_observed_adjacency(self):
        return self.mask_structure == "fixed-chain" or bool(self.has_frozen_adjacency.item())

    def get_raw_observed_adjacency(self):
        if self.mask_structure == "fixed-chain":
            return self.fixed_adjacency
        if bool(self.has_frozen_adjacency.item()):
            return self.frozen_adjacency
        weights = T.sigmoid(self.edge_logits)
        adjacency = weights.new_zeros(3, 3)
        adjacency[0, 1] = weights[0]  # x -> y
        adjacency[1, 0] = weights[1]  # y -> x
        adjacency[1, 2] = weights[2]  # y -> z
        adjacency[2, 1] = weights[3]  # z -> y
        return adjacency

    def project_adjacency_to_order(self, adjacency, order):
        if len(order) != 3:
            raise ValueError("order must contain exactly three variables.")
        ranks = {node: idx for idx, node in enumerate(order)}
        projected = adjacency.new_zeros(adjacency.shape)

        if ranks[0] < ranks[1]:
            projected[0, 1] = adjacency[0, 1]
        else:
            projected[1, 0] = adjacency[1, 0]

        if ranks[1] < ranks[2]:
            projected[1, 2] = adjacency[1, 2]
        else:
            projected[2, 1] = adjacency[2, 1]

        return projected

    def freeze_to_order(self, order):
        if self.mask_structure != "learnable":
            return
        raw_adjacency = self.get_raw_observed_adjacency().detach()
        projected = self.project_adjacency_to_order(raw_adjacency, order)
        self.frozen_adjacency.copy_(projected.to(device=self.frozen_adjacency.device, dtype=self.frozen_adjacency.dtype))
        self.has_frozen_adjacency.fill_(True)

    def get_observed_adjacency(self):
        adjacency = self.get_raw_observed_adjacency()
        if self.adjacency_normalization == "sum":
            adjacency = adjacency / adjacency.sum().clamp_min(self.gate_renorm_eps)
        return adjacency

    def expanded_gate_matrix(self):
        adjacency = self.get_observed_adjacency()
        gate_edges = adjacency
        if not self._uses_hard_observed_adjacency():
            gate_edges = self.gate_floor + (1.0 - self.gate_floor) * adjacency

        gates = adjacency.new_zeros(self.seq_len, self.seq_len)
        gates.fill_diagonal_(1.0)

        # u_x can attend to Y via y -> x
        gates[0, 4] = gate_edges[1, 0]
        # u_y can attend to X via x -> y and Z via z -> y
        gates[1, 3] = gate_edges[0, 1]
        gates[1, 5] = gate_edges[2, 1]
        # u_z can attend to Y via y -> z
        gates[2, 4] = gate_edges[1, 2]
        return gates

    def expanded_mask_weights(self):
        return self.expanded_gate_matrix()

    def apply_hard_mask(self, logits):
        hard_mask = self.hard_mask.to(device=logits.device, dtype=logits.dtype)
        return logits + hard_mask.unsqueeze(0).unsqueeze(0)

    def apply_graph_gates(self, attn_weights):
        gates = self.expanded_gate_matrix().to(device=attn_weights.device, dtype=attn_weights.dtype)
        gated = attn_weights * gates.unsqueeze(0).unsqueeze(0)
        return gated / (gated.sum(dim=-1, keepdim=True) + self.gate_renorm_eps)

    def observed_dependency_scores(self, num_variables):
        if num_variables * 2 != self.seq_len:
            raise ValueError("num_variables does not match the configured sequence length.")
        # Return target <- source scores for sampling order heuristics.
        return self.get_observed_adjacency().transpose(0, 1)

    def dag_penalty(self):
        if self._uses_hard_observed_adjacency():
            return T.zeros((), device=self.hard_mask.device, dtype=T.float32)
        adjacency = self.get_observed_adjacency()
        gram = adjacency * adjacency
        system = self.dagma_s * T.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) - gram
        sign, logabsdet = T.linalg.slogdet(system)
        if T.any(sign <= 0) or not T.isfinite(logabsdet):
            raise ValueError("DAG penalty is undefined because sI - W o W left the positive-logdet domain.")
        return -logabsdet + gram.shape[0] * math.log(self.dagma_s)

    def l1_penalty(self):
        if self._uses_hard_observed_adjacency():
            return T.zeros((), device=self.hard_mask.device, dtype=T.float32)
        return self.get_raw_observed_adjacency().abs().sum()


class MaskedAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.0, residual=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.residual = residual

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )

    def _reshape_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, mask_module):
        residual = x
        x_norm = self.norm1(x)

        q = self._reshape_heads(self.q_proj(x_norm))
        k = self._reshape_heads(self.k_proj(x_norm))
        v = self._reshape_heads(self.v_proj(x_norm))

        attn_logits = T.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_logits = mask_module.apply_hard_mask(attn_logits)
        attn_weights = T.softmax(attn_logits, dim=-1)
        attn_weights = mask_module.apply_graph_gates(attn_weights)
        attn_weights = self.dropout(attn_weights)

        attn_out = T.matmul(attn_weights, v)
        attn_out = self._merge_heads(attn_out)
        attn_out = self.dropout(self.out_proj(attn_out))
        if self.residual:
            x = residual + attn_out
        else:
            x = attn_out

        ffn_out = self.dropout(self.ffn(self.norm2(x)))
        if self.residual:
            x = x + ffn_out
        else:
            x = ffn_out
        return x


class PAGModel(nn.Module):
    def __init__(
        self,
        num_variables,
        latent_dim=8,
        embed_dim=64,
        num_heads=4,
        num_layers=1,
        ffn_hidden_dim=128,
        latent_mlp_hidden_dim=128,
        latent_mlp_layers=2,
        head_hidden_dim=64,
        graph="chain",
        mask_structure="learnable",
        dagma_s=2.0,
        init_edge_logit=0,
        mask_init_std=1.0,
        adjacency_normalization="none",
        gate_floor=0.05,
        gate_renorm_eps=1e-6,
        dropout=0.0,
        latent_prior="normal",
        residual=True,
    ):
        super().__init__()
        if graph != "chain":
            raise ValueError("Only graph='chain' is supported right now.")
        if num_variables <= 0:
            raise ValueError("num_variables must be positive.")
        if num_variables != 3:
            raise ValueError("The chain PAG model requires exactly three variables: X, Y, Z.")

        self.num_variables = num_variables
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.latent_prior = latent_prior
        self.graph = graph
        self.mask_structure = mask_structure
        self.dagma_s = dagma_s
        self.init_edge_logit = init_edge_logit
        self.adjacency_normalization = adjacency_normalization
        self.gate_floor = gate_floor
        self.gate_renorm_eps = gate_renorm_eps

        self.variable_embedding = nn.Embedding(num_variables, embed_dim)
        self.token_type_embedding = nn.Embedding(2, embed_dim)
        self.value_embeddings = nn.ModuleList([nn.Embedding(2, embed_dim) for _ in range(num_variables)])
        self.latent_embedders = nn.ModuleList([
            SimpleMLP(latent_dim, embed_dim, latent_mlp_hidden_dim, num_layers=latent_mlp_layers)
            for _ in range(num_variables)
        ])

        self.mask_module = LearnableDAGChainAttentionMask(
            seq_len=2 * num_variables,
            graph=graph,
            mask_structure=mask_structure,
            dagma_s=dagma_s,
            init_edge_logit=init_edge_logit,
            init_edge_logit_std=mask_init_std,
            adjacency_normalization=adjacency_normalization,
            gate_floor=gate_floor,
            gate_renorm_eps=gate_renorm_eps,
        )

        self.blocks = nn.ModuleList([
            MaskedAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, head_hidden_dim),
                nn.GELU(),
                nn.Linear(head_hidden_dim, 1),
            )
            for _ in range(num_variables)
        ])

    @staticmethod
    def _sample_from_logits(logits):
        probs = T.sigmoid(logits)
        return T.bernoulli(probs).float()

    def sample_latents(self, batch_size, mc_samples, device):
        shape = (batch_size, mc_samples, self.num_variables, self.latent_dim)
        if self.latent_prior == "normal":
            return T.randn(shape, device=device)
        if self.latent_prior == "uniform":
            return T.rand(shape, device=device)
        raise ValueError("Unsupported latent_prior '{}'.".format(self.latent_prior))

    def _embed_latents(self, latents):
        latent_tokens = []
        for idx, embedder in enumerate(self.latent_embedders):
            cur_latent = latents[:, :, idx, :]
            cur_embed = embedder(cur_latent)
            latent_tokens.append(cur_embed.unsqueeze(2))
        return T.cat(latent_tokens, dim=2)

    def _embed_values(self, values, mc_samples):
        value_tokens = []
        for idx, embedding in enumerate(self.value_embeddings):
            cur_value = values[:, idx].long()
            cur_embed = embedding(cur_value).unsqueeze(1).expand(-1, mc_samples, -1)
            value_tokens.append(cur_embed.unsqueeze(2))
        return T.cat(value_tokens, dim=2)

    def _add_token_metadata(self, tokens):
        device = tokens.device
        variable_ids = T.arange(self.num_variables, device=device)
        variable_embed = self.variable_embedding(variable_ids).unsqueeze(0).unsqueeze(0)
        variable_embed = variable_embed.expand(tokens.shape[0], tokens.shape[1], -1, -1)

        token_type_ids = T.tensor([0] * self.num_variables + [1] * self.num_variables, device=device)
        token_type_embed = self.token_type_embedding(token_type_ids).unsqueeze(0).unsqueeze(0)
        token_type_embed = token_type_embed.expand(tokens.shape[0], tokens.shape[1], -1, -1)

        variable_embed = T.cat([variable_embed, variable_embed], dim=2)
        return tokens + variable_embed + token_type_embed

    def _compute_logits_from_u_tokens(self, u_tokens):
        logits = []
        for idx, head in enumerate(self.output_heads):
            cur_logits = head(u_tokens[:, :, idx, :]).squeeze(-1)
            logits.append(cur_logits.unsqueeze(-1))
        return T.cat(logits, dim=-1)

    def forward(self, values, latents):
        batch_size, mc_samples, _, _ = latents.shape

        latent_tokens = self._embed_latents(latents)
        value_tokens = self._embed_values(values, mc_samples)
        tokens = T.cat([latent_tokens, value_tokens], dim=2)
        tokens = self._add_token_metadata(tokens)

        tokens = tokens.view(batch_size * mc_samples, 2 * self.num_variables, self.embed_dim)
        for block in self.blocks:
            tokens = block(tokens, self.mask_module)
        tokens = tokens.view(batch_size, mc_samples, 2 * self.num_variables, self.embed_dim)

        u_tokens = tokens[:, :, :self.num_variables, :]
        logits = self._compute_logits_from_u_tokens(u_tokens)
        return logits

    def conditional_log_prob(self, values, latents):
        logits = self.forward(values, latents)
        expanded_values = values.unsqueeze(1).expand(-1, latents.shape[1], -1)
        log_prob = -F.binary_cross_entropy_with_logits(logits, expanded_values, reduction='none')
        return log_prob.sum(dim=-1)

    def marginal_log_prob(self, values, mc_samples, latents=None):
        if latents is None:
            latents = self.sample_latents(values.shape[0], mc_samples, values.device)
        cond_log_prob = self.conditional_log_prob(values, latents)
        return T.logsumexp(cond_log_prob, dim=1) - math.log(cond_log_prob.shape[1])

    def get_observed_adjacency(self):
        return self.mask_module.get_observed_adjacency()

    def expanded_gate_matrix(self):
        return self.mask_module.expanded_gate_matrix()

    def expanded_mask_weights(self):
        return self.mask_module.expanded_mask_weights()

    def dag_penalty(self):
        return self.mask_module.dag_penalty()

    def l1_penalty(self):
        return self.mask_module.l1_penalty()

    def _sampling_dependency_scores(self):
        return self.mask_module.observed_dependency_scores(self.num_variables)

    def _build_confidence_order(self):
        dep_scores = self._sampling_dependency_scores().clone()
        dep_scores.fill_diagonal_(0.0)

        remaining = set(range(self.num_variables))
        order = []

        while remaining:
            available = []
            for idx in remaining:
                incoming = dep_scores[idx, list(remaining)].sum().item()
                if incoming <= 1e-8:
                    total_dep = dep_scores[idx].sum().item()
                    available.append((total_dep, idx))

            if available:
                _, chosen = min(available)
            else:
                ranked = []
                for idx in remaining:
                    incoming = dep_scores[idx, list(remaining)].sum().item()
                    total_dep = dep_scores[idx].sum().item()
                    ranked.append((incoming, total_dep, idx))
                _, _, chosen = min(ranked)

            order.append(chosen)
            remaining.remove(chosen)

        return order

    def sample_chain(self, sample_count, depth, device):
        if self.graph != "chain":
            raise ValueError("Only graph='chain' is supported right now.")
        if depth != 3:
            raise ValueError("The chain PAG sampler requires depth=3.")

        latents = self.sample_latents(sample_count, 1, device=device)
        values = T.zeros((sample_count, self.num_variables), device=device)

        # Build one order from the mask and sample variables once in that order.
        # When the mask becomes learned later, the same routine can rank variables
        # by dependency strength and fall back gracefully if the mask is cyclic.
        update_order = self._build_confidence_order()
        if len(update_order) != depth:
            raise ValueError("The chain PAG sampler order must contain exactly three variables.")

        for coord in update_order:
            logits = self.forward(values, latents).squeeze(1)
            coord_logits = logits[:, coord:coord + 1]
            values[:, coord] = self._sample_from_logits(coord_logits).squeeze(1)

        return values
