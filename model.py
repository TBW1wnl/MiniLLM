"""
A minimal GPT: a decoder-only Transformer, written to be read top to bottom.

The whole model answers one question: "given the tokens so far, what is the
probability of each possible next token?" Everything below is machinery to
compute that distribution well.

Data flow for a batch of B sequences of T tokens, with embedding width C:

    token ids (B, T)
      -> token embedding + position embedding        (B, T, C)
      -> n_layer x Block [attention, then MLP]        (B, T, C)
      -> final LayerNorm
      -> lm_head: one score (logit) per vocab entry   (B, T, vocab_size)

Every position t produces a prediction for token t+1, so one forward pass
over T tokens gives T training examples at once.

With n_experts > 0, the MLP of every block is replaced by a Mixture of
Experts (see the MoE class): more parameters, same compute per token.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # maximum context length, in tokens
    n_layer: int = 6  # number of Transformer blocks stacked on top of each other
    n_head: int = 6  # attention heads per block
    n_embd: int = 384  # width of the vector that represents each token
    dropout: float = 0.2
    flash: bool = True  # fused attention kernel; False runs the explicit math below
    n_experts: int = 0  # 0 = plain MLP in every block, > 0 = Mixture of Experts
    experts_per_token: int = 2  # how many experts each token is routed to (MoE only)


class CausalSelfAttention(nn.Module):
    """
    Lets every token gather information from the tokens before it.

    Each token emits three vectors:
      query -- "what am I looking for?"
      key   -- "what do I contain?"
      value -- "what do I hand over if someone attends to me?"
    The attention weight from token i to token j is softmax(q_i . k_j), and
    token i's output is the weighted sum of the values. "Causal" means token i
    may only look at j <= i: otherwise it could peek at the answer it is
    supposed to predict.

    Multi-head: the C dimensions are split into n_head independent heads, so
    the block can track several relationships at once (e.g. one head follows
    the speaker's name, another the end of the current line).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.flash = config.flash
        self.dropout = config.dropout

        # One matrix computes queries, keys and values for all heads at once.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Mixes the heads' outputs back together.
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular boolean mask, only used by the explicit path.
        # persistent=False keeps it out of checkpoints (it is not learned).
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim): each head gets its own slice.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            # Same result as the else-branch, computed by a fused GPU kernel
            # that never materializes the (T, T) attention matrix.
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            # Similarity of every query with every key: (B, nh, T, T).
            # Dividing by sqrt(head_dim) keeps the scores from growing with
            # the head size, which would make the softmax too peaky.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Forbid looking at the future: -inf becomes 0 after softmax.
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)  # each row now sums to 1
            att = self.attn_dropout(att)
            y = att @ v  # weighted sum of values: (B, nh, T, head_dim)

        # Put the heads side by side again: (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """
    Per-token feed-forward network. Attention moves information *between*
    tokens; the MLP then processes it *within* each token. The 4x expansion
    is the convention from the original Transformer paper.
    """

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class MoE(nn.Module):
    """
    Mixture of Experts: n_experts independent MLPs ("experts") plus a small
    router that picks, for each token, the experts_per_token best ones.

    With 8 experts and 2 active, the layer holds 8x the parameters of one MLP
    but each token only pays for 2 of them. That is the whole point of MoE:
    grow the model's capacity (what it can store) without growing the compute
    per token (what it costs to run).

        token vector x
          -> router: one score per expert -> softmax -> keep the top 2
          -> output = w1 * expert_a(x) + w2 * expert_b(x)
             where w1, w2 are the two kept probabilities, rescaled to sum to 1

    Left alone, the router tends to collapse: a few experts get slightly
    better early, receive all the tokens, get even better, and the rest are
    never trained. The auxiliary load-balancing loss returned by forward()
    pushes the router to spread tokens evenly.
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.k = config.experts_per_token
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])
        # Fraction of routing slots each expert received in the last forward
        # pass, kept only to be printed during training.
        self.last_load = None

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # routing is per token: forget the (B, T) layout

        # Router probabilities, in float32 so the softmax is not distorted by
        # bfloat16 rounding.
        probs = F.softmax(self.router(x_flat).float(), dim=-1)  # (N, n_experts)
        weights, chosen = probs.topk(self.k, dim=-1)  # (N, k) each
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Run each expert only on the tokens that chose it, then scatter its
        # weighted output back to those tokens' rows. A plain loop over the
        # experts is the readable version; production code sorts tokens by
        # expert and runs all experts in one grouped matmul instead.
        out = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (chosen == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y = expert(x_flat[token_idx]) * weights[token_idx, slot, None]
            out.index_add_(0, token_idx, y.to(out.dtype))

        # Load-balancing loss (Switch Transformer, Fedus et al. 2021):
        #   load[e]       = fraction of routing slots that went to expert e
        #   importance[e] = average router probability given to expert e
        #   aux = n_experts * sum_e load[e] * importance[e]
        # It equals 1 when both are uniform and grows when the router favours
        # a few experts. load itself comes from topk and has no gradient; the
        # gradient flows through importance.
        load = F.one_hot(chosen, self.n_experts).float().sum(dim=1).mean(dim=0) / self.k
        importance = probs.mean(dim=0)
        aux_loss = self.n_experts * (load * importance).sum()
        self.last_load = load.detach()

        return out.view(B, T, C), aux_loss


class Block(nn.Module):
    """
    One Transformer block: communicate (attention), then compute (MLP).

    The "x = x + ..." residual connections are what make deep stacks
    trainable: each block only learns a correction to its input, and the
    gradient has a direct path back to the first layers. LayerNorm is applied
    *before* each sub-layer ("pre-norm"), which is more stable than the
    original post-norm design.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MoE(config) if config.n_experts > 0 else MLP(config)

    def forward(self, x):
        """Returns the new x and this block's load-balancing loss (0 without MoE)."""
        x = x + self.attn(self.ln1(x))
        if isinstance(self.mlp, MoE):
            y, aux_loss = self.mlp(self.ln2(x))
        else:
            y, aux_loss = self.mlp(self.ln2(x)), x.new_zeros(())
        return x + y, aux_loss


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding: a learned vector for each vocab entry ("what").
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding: a learned vector for each slot 0..block_size-1
        # ("where"). Without it, attention would see the context as an
        # unordered bag of tokens.
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and the output projection share
        # one matrix. Both map between "token id" and "vector" space, so it
        # saves parameters and usually helps a little.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream,
        # so its variance does not grow with the number of blocks.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self):
        # Position embeddings are excluded by convention; the tied lm_head is
        # counted once because parameters() deduplicates shared tensors.
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def num_active_params(self):
        """Parameters actually used for one token: with MoE, only k experts run."""
        cfg = self.config
        if cfg.n_experts == 0:
            return self.num_params()
        per_expert = sum(p.numel() for p in self.blocks[0].mlp.experts[0].parameters())
        return self.num_params() - cfg.n_layer * (cfg.n_experts - cfg.experts_per_token) * per_expert

    def forward(self, idx, targets=None):
        """
        Returns (logits, loss, aux_loss). loss is the cross-entropy (None
        without targets); aux_loss is the MoE load-balancing loss averaged over
        the blocks (0 for a dense model). They are kept apart so that the
        reported loss stays comparable between dense and MoE models.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence of {T} tokens > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        aux_loss = 0.0
        for block in self.blocks:
            x, block_aux = block(x)
            aux_loss = aux_loss + block_aux
        aux_loss = aux_loss / len(self.blocks)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy = -log(probability given to the correct next
            # token), averaged over all B*T positions. A model guessing
            # uniformly over 65 chars scores ln(65) ~= 4.17.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, aux_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive sampling: predict one token, append it, repeat.
        Yields each new token as a (B, 1) tensor so callers can stream it.

        temperature < 1 sharpens the distribution (safer, more repetitive),
        > 1 flattens it (more creative, more mistakes).
        top_k keeps only the k most likely tokens before sampling.
        """
        for _ in range(max_new_tokens):
            # The model has never seen positions beyond block_size, so only
            # the most recent block_size tokens are fed back in.
            context = idx[:, -self.config.block_size :]
            logits, _, _ = self(context)
            logits = logits[:, -1, :] / temperature  # only the last position matters
            if top_k is not None:
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth_best] = float("-inf")
            probs = F.softmax(logits.float(), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            yield next_id
