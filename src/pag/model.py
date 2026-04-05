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


class MixedChainAttentionMask(nn.Module):
    def __init__(self, num_heads, seq_len, graph="chain", mask_mode="additive"):
        super().__init__()
        if graph != "chain":
            raise ValueError("Only graph='chain' is supported right now.")
        if seq_len != 6:
            raise ValueError("The chain mask only supports three PAG variables X, Y, Z.")
        if mask_mode not in {"additive", "multiplicative"}:
            raise ValueError("mask mode must be 'additive' or 'multiplicative'.")

        self.mask_mode = mask_mode
        self.num_heads = num_heads
        self.seq_len = seq_len

        # Token order is fixed as [u_x, u_y, u_z, X, Y, Z].
        # Reverse chain factorization:
        # Z depends on u_z
        # Y depends on (u_y, Z)
        # X depends on (u_x, Y)
        allow = T.zeros(seq_len, seq_len, dtype=T.bool)
        allow[0, 0] = True
        allow[0, 4] = True
        allow[1, 1] = True
        allow[1, 5] = True
        allow[2, 2] = True
        allow[3, 3] = True
        allow[4, 4] = True
        allow[5, 5] = True
        self.register_buffer('allow_mask', allow.unsqueeze(0).expand(num_heads, -1, -1))

    def forward(self, logits):
        if self.mask_mode == "additive":
            blocked = T.full_like(logits, -1e9)
            return T.where(self.allow_mask.unsqueeze(0), logits, blocked)

        return T.where(self.allow_mask.unsqueeze(0), logits, T.full_like(logits, -1e9))

    def observed_dependency_scores(self, num_variables):
        if num_variables * 2 != self.seq_len:
            raise ValueError("num_variables does not match the configured sequence length.")
        allow = self.allow_mask.float().mean(dim=0)
        return allow[:num_variables, num_variables:]


class MaskedAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, graph="chain", mask_mode="additive",
                 dropout=0.0, residual=True):
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

        self.mask_module = None
        self.graph = graph
        self.mask_mode = mask_mode
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

    def set_seq_len(self, seq_len):
        self.mask_module = MixedChainAttentionMask(
            self.num_heads,
            seq_len,
            graph=self.graph,
            mask_mode=self.mask_mode,
        )

    def _reshape_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x):
        if self.mask_module is None:
            self.set_seq_len(x.shape[1])

        residual = x
        x_norm = self.norm1(x)

        q = self._reshape_heads(self.q_proj(x_norm))
        k = self._reshape_heads(self.k_proj(x_norm))
        v = self._reshape_heads(self.v_proj(x_norm))

        attn_logits = T.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_logits = self.mask_module(attn_logits)
        attn_weights = T.softmax(attn_logits, dim=-1)
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
        mask_mode="additive",
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

        self.variable_embedding = nn.Embedding(num_variables, embed_dim)
        self.token_type_embedding = nn.Embedding(2, embed_dim)
        self.value_embeddings = nn.ModuleList([nn.Embedding(2, embed_dim) for _ in range(num_variables)])
        self.latent_embedders = nn.ModuleList([
            SimpleMLP(latent_dim, embed_dim, latent_mlp_hidden_dim, num_layers=latent_mlp_layers)
            for _ in range(num_variables)
        ])

        self.blocks = nn.ModuleList([
            MaskedAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                graph=graph,
                mask_mode=mask_mode,
                dropout=dropout,
                residual=residual,
            )
            for _ in range(num_layers)
        ])
        seq_len = 2 * num_variables
        for block in self.blocks:
            block.set_seq_len(seq_len)

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
            tokens = block(tokens)
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

    def _sampling_dependency_scores(self):
        dep_scores = []
        for block in self.blocks:
            if block.mask_module is None:
                block.set_seq_len(2 * self.num_variables)
            dep_scores.append(block.mask_module.observed_dependency_scores(self.num_variables))
        if not dep_scores:
            raise ValueError("At least one attention block is required to build a sampling order.")
        return T.stack(dep_scores, dim=0).mean(dim=0)

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
