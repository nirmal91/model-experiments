"""
GRPO — Group Relative Policy Optimization, from scratch.

The RL algorithm behind DeepSeek-R1 (introduced in DeepSeekMath, used
unchanged in R1). The idea, versus PPO:

  PPO needs a learned critic (value network) to compute advantages.
  GRPO deletes the critic. Instead, for each prompt it samples a GROUP of
  G completions and uses the group's own reward statistics as the baseline:

      A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

  Completions better than their siblings get pushed up; worse get pushed
  down. The group mean plays the role of the value function.

Per-token objective (maximized):

      J = E_i,t [ min( rho_t * A_i, clip(rho_t, 1-eps, 1+eps) * A_i )
                  - beta * kl_t ]

  where rho_t = pi_theta(o_t | q, o_<t) / pi_old(o_t | q, o_<t), and kl_t is
  the k3 estimator of KL(pi_theta || pi_ref) used by the paper:

      kl_t = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1   (always >= 0)

  pi_ref is a frozen snapshot (the base model for R1-Zero, the SFT model for
  R1 stage 2) — it stops the policy from drifting into unreadable degenerate
  outputs while it chases reward.

This file is generic over the task: it takes a `reward_fn(completion_text,
ground_truth)` and operates on any GPT + char tokenizer from experiment 1.
"""
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    group_size: int = 8          # G: completions sampled per prompt
    prompts_per_step: int = 8    # questions per RL step
    max_new_tokens: int = 80
    clip_eps: float = 0.2        # PPO-style ratio clip
    kl_beta: float = 0.02        # weight of KL(pi_theta || pi_ref)
    inner_epochs: int = 2        # mu: gradient updates per sampled batch
    lr: float = 3e-4
    temperature: float = 1.0
    total_steps: int = 200
    log_every: int = 5
    seed: int = 0


class GRPOTrainer:
    def __init__(self, policy, ref_model, tok, make_problem, reward_fn,
                 cfg: GRPOConfig, eos_char='\n'):
        self.policy = policy
        self.ref = ref_model
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.tok = tok
        self.make_problem = make_problem   # rng -> (prompt_str, ground_truth)
        self.reward_fn = reward_fn         # (completion_str, truth) -> (r, correct, fmt)
        self.cfg = cfg
        self.eos_id = tok.stoi[eos_char]
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        self.rng = random.Random(cfg.seed)
        self.history = []

    # ---------------- sampling ----------------

    @torch.no_grad()
    def _sample_group_batch(self):
        """Sample G completions for each of N prompts in one batched pass.
        Returns prompts, truths, and a (N*G, T) tensor of full sequences."""
        cfg = self.cfg
        problems = [self.make_problem(self.rng) for _ in range(cfg.prompts_per_step)]
        prompts = [p for p, _ in problems]
        truths = [t for _, t in problems]

        # all prompts are fixed-width, so they stack into one tensor;
        # repeat each prompt G times -> one big generation batch
        prompt_ids = torch.tensor(
            [self.tok.encode(p) for p in prompts], dtype=torch.long)
        idx = prompt_ids.repeat_interleave(cfg.group_size, dim=0)  # (N*G, P)
        P = idx.size(1)

        self.policy.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool)
        for _ in range(cfg.max_new_tokens):
            idx_cond = idx[:, -self.policy.config.block_size:]
            logits, _ = self.policy(idx_cond)
            logits = logits[:, -1, :] / cfg.temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            # once a row has emitted EOS, keep padding it with EOS
            nxt[finished] = self.eos_id
            idx = torch.cat((idx, nxt), dim=1)
            finished |= (nxt.squeeze(1) == self.eos_id)
            if finished.all():
                break
        self.policy.train()
        return prompts, truths, idx, P

    def _decode_completion(self, row, P):
        """Text after the prompt, truncated at EOS."""
        toks = row[P:].tolist()
        if self.eos_id in toks:
            toks = toks[:toks.index(self.eos_id)]
        return self.tok.decode(toks)

    # ---------------- scoring ----------------

    def _completion_mask(self, seqs, P):
        """(B, T-1) mask over *predicted* positions: 1 for completion tokens
        up to and including the first EOS, 0 for prompt and post-EOS padding."""
        B, T = seqs.shape
        targets = seqs[:, 1:]
        mask = torch.zeros(B, T - 1)
        mask[:, P - 1:] = 1.0
        is_eos = (targets == self.eos_id).float()
        # cumulative EOS count *before* each position; >0 means past first EOS
        past_eos = (is_eos.cumsum(dim=1) - is_eos) > 0
        mask[past_eos] = 0.0
        return mask

    @staticmethod
    def _token_logprobs(model, seqs):
        """Log-prob of each realized next token: (B, T-1)."""
        logits, _ = model(seqs[:, :-1])
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)

    # ---------------- one RL step ----------------

    def step(self, step_idx: int):
        cfg = self.cfg
        prompts, truths, seqs, P = self._sample_group_batch()
        B = seqs.size(0)  # N * G

        # rule-based rewards per completion
        rewards, corrects, fmts, lengths = [], [], [], []
        for i in range(B):
            text = self._decode_completion(seqs[i], P)
            truth = truths[i // cfg.group_size]
            r, c, f = self.reward_fn(text, truth)
            rewards.append(r); corrects.append(c); fmts.append(f)
            lengths.append(len(text))
        rewards = torch.tensor(rewards)

        # group-relative advantage: normalize within each prompt's group
        r_grouped = rewards.view(cfg.prompts_per_step, cfg.group_size)
        adv = (r_grouped - r_grouped.mean(dim=1, keepdim=True)) \
            / (r_grouped.std(dim=1, keepdim=True) + 1e-6)
        adv = adv.view(B, 1)  # broadcast over tokens (outcome reward -> same
        #                       advantage for every token of the completion)

        mask = self._completion_mask(seqs, P)
        with torch.no_grad():
            logp_old = self._token_logprobs(self.policy, seqs)
            logp_ref = self._token_logprobs(self.ref, seqs)

        # mu inner epochs of the clipped surrogate on this sampled batch
        for _ in range(cfg.inner_epochs):
            logp_new = self._token_logprobs(self.policy, seqs)
            ratio = (logp_new - logp_old).exp()
            surr = torch.min(ratio * adv,
                             ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
            # k3 KL estimator: pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
            log_rr = logp_ref - logp_new
            kl = log_rr.exp() - log_rr - 1
            per_tok = surr - cfg.kl_beta * kl
            # mean over completion tokens, then over the batch (paper's 1/|o_i| sum)
            loss = -((per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

        stats = {
            'step': step_idx,
            'reward': rewards.mean().item(),
            'accuracy': sum(corrects) / B,
            'format': sum(fmts) / B,
            'resp_len': sum(lengths) / B,
            'kl': (kl.detach() * mask).sum().item() / mask.sum().clamp(min=1).item(),
        }
        self.history.append(stats)
        return stats

    def train(self, sample_logger=None):
        for s in range(self.cfg.total_steps):
            stats = self.step(s)
            if s % self.cfg.log_every == 0 or s == self.cfg.total_steps - 1:
                print(f"step {stats['step']:4d} | reward {stats['reward']:.3f} | "
                      f"acc {stats['accuracy']:.2f} | fmt {stats['format']:.2f} | "
                      f"len {stats['resp_len']:.0f} | kl {stats['kl']:.4f}", flush=True)
                if sample_logger:
                    sample_logger(self, s)
        return self.history
