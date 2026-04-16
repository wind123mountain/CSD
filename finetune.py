import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from data_utils.data_utils import LLMDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics

from peft import PeftModel

torch.set_num_threads(4)

NNM_CFG = dict(
    # -- NNM core --
    lambda_nnm        = 0.10,       # NNM loss weight
    nnm_warmup        = 100,        # warmup steps for NNM weight
    K_centroids       = 128,        # number of running centroids per layer
    d_prime           = 256,        # random projection dimension
    eta_centroid      = 0.05,       # centroid EMA rate
    T_dead            = 50,         # steps before dead centroid is revived
    sigma_layer       = 0.15,       # Gaussian layer weighting sigma
    n_mid_layers      = 4,          # number of mid layers to align
    ns_iters          = 5,          # Newton-Schulz iterations
 
    # -- Teacher correction --
    do_teacher_correction = True,
    tc_lambda         = 0.10,
    tc_steps          = 1,
 
    # -- Token filtering (STAPO + Beyond 80/20) --
    s2t_tau_p         = None,       # spurious token prob threshold
    s2t_tau_h         = None,        # spurious token entropy threshold
    high_ent_rho      = 0.2,        # keep top 20% entropy tokens for KL
 
    # -- Top-K logit KD --
    top_k_logits      = 20,
 
    # -- Difficulty-aware weighting --
    use_difficulty_weight  = False,
    difficulty_early_layer = 2,
 
    # -- Temperature annealing --
    kl_temp_max       = 3.0,
    kl_temp_min       = 1.0,
 
    # -- Centroid pre-pass --
    centroid_prepass_batches = 3000,
)

SEED = 42

_NS_COEFFS = (15 / 8, -10 / 8, 3 / 8)
 
 
def newton_schulz_polar(M, n_iters=5):
    assert M.ndim == 2
    dtype = M.dtype
    transposed = False
    if M.shape[0] < M.shape[1]:
        M = M.T; transposed = True
    X = M / (M.norm() + 1e-7)
    a, b, c = _NS_COEFFS
    for _ in range(n_iters):
        A = X.T @ X
        X = a * X + b * (X @ A) + c * (X @ (A @ A))
    if transposed:
        X = X.T
    return X.to(dtype)
 
 
class _NuclearNormNS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, n_iters):
        with torch.no_grad():
            P = newton_schulz_polar(M.detach(), n_iters)
        ctx.save_for_backward(P)
        return (P * M).sum()
 
    @staticmethod
    def backward(ctx, grad_output):
        (P,) = ctx.saved_tensors
        return grad_output * P, None
 
 
def nuclear_norm_ns(M, n_iters=5):
    return _NuclearNormNS.apply(M, n_iters)
 
 
# ============================================================================
#  RUNNING CENTROIDS (from NNM-KD)
# ============================================================================
class RunningCentroids:
    def __init__(self, K, d, eta, T_dead, device):
        self.K, self.d, self.eta, self.T_dead = K, d, eta, T_dead
        self.device = device
        self.C    = torch.randn(K, d, device=device, dtype=torch.float32) * 0.01
        self.dead = torch.zeros(K, device=device, dtype=torch.int32)
        self._step = 0
 
    @torch.no_grad()
    def update(self, H):
        H = H.to(self.device).float()
        if H.shape[0] == 0:
            return
        self._step += 1
        eta = self.eta / (1 + 0.001 * self._step)
        dists  = torch.cdist(H, self.C)
        assign = dists.argmin(dim=1)
        for k in range(self.K):
            mask = (assign == k)
            if mask.any():
                self.C[k] = (1 - eta) * self.C[k] + eta * H[mask].mean(0)
                self.dead[k] = 0
            else:
                self.dead[k] += 1
                if self.dead[k] >= self.T_dead:
                    self.C[k] = H[random.randint(0, len(H) - 1)].clone()
                    self.dead[k] = 0
 
 
# ============================================================================
#  HIDDEN PROJECTOR — 2-layer MLP (from NNM-KD v3)
# ============================================================================
class HiddenProjector(nn.Module):
    def __init__(self, d_teacher, d_student):
        super().__init__()
        d_mid = d_teacher // 2
        self.net = nn.Sequential(
            nn.Linear(d_teacher, d_mid, bias=False),
            nn.GELU(),
            nn.Linear(d_mid, d_student, bias=False),
        )
        nn.init.orthogonal_(self.net[2].weight)
 
    def forward(self, x):
        return self.net(x.float())
 
 
# ============================================================================
#  NNM UTILITIES (from NNM-KD)
# ============================================================================
def make_R(d, d_prime, device):
    torch.manual_seed(SEED)
    return (torch.randn(d, d_prime, device=device) / math.sqrt(d_prime)).float()
 
 
def layer_weight(l, L, sigma=0.15):
    return math.exp(-((l / L - 0.5) ** 2) / (2 * sigma ** 2))
 
 
def select_mid_layers(n_layers, n_mid):
    """Shifted to 40%-85% (later layers encode reasoning)."""
    lo = max(1, int(0.40 * n_layers))
    hi = min(n_layers, int(0.85 * n_layers))
    if lo >= hi:
        lo = max(0, hi - n_mid)
    return sorted(set(int(i) for i in np.linspace(lo, hi, n_mid, dtype=int).tolist()))
 
 
def nnm_loss_one_layer(H_s, H_t_proj, C_s, C_t, R, lw, ns_iters):
    H_s = H_s.float(); H_t_proj = H_t_proj.float().detach()
    C_s = C_s.float().detach(); C_t = C_t.float().detach(); R = R.float()
    M_s = torch.cat([C_s, H_s], dim=0) @ R
    M_t = torch.cat([C_t, H_t_proj], dim=0) @ R
    m, n = M_s.shape; scale = math.sqrt(m * n)
    nn_s = nuclear_norm_ns(M_s, ns_iters) / scale
    nn_t = (nuclear_norm_ns(M_t, ns_iters) / scale).detach()
    return lw * (nn_s - nn_t) ** 2
 
 
@torch.no_grad()
def correct_teacher_hiddens(H_T, C_T, R, lam, ns_iters, tc_steps=1):
    H = H_T.float().clone(); K = C_T.shape[0]; R = R.float()
    for _ in range(tc_steps):
        M0 = torch.cat([C_T.float(), H], dim=0) @ R
        P = newton_schulz_polar(M0, ns_iters)
        G_X = P[K:] @ R.T
        H = H + lam * G_X
    return H.to(H_T.dtype)
 
 
# ============================================================================
#  FORWARD WITH HIDDENS (from NNM-KD)
# ============================================================================
def forward_with_hiddens(model, input_ids, attention_mask, layer_ids,
                         target_device, no_grad=False, label_mask=None):
    import contextlib
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True, use_cache=False)
    logits = out.logits.to(target_device)
    hiddens_active, hiddens_full = {}, {}
    for lid in layer_ids:
        h = out.hidden_states[lid].to(target_device).float()
        hiddens_full[lid] = h
        flat_h = h.reshape(-1, h.shape[-1])
        if label_mask is not None:
            flat_mask = label_mask.to(target_device).reshape(-1).bool()
        else:
            flat_mask = attention_mask.to(target_device).reshape(-1).bool()
        ha = flat_h[flat_mask]
        hiddens_active[lid] = ha.detach() if no_grad else ha
    return hiddens_active, hiddens_full, logits
 
 
# ============================================================================
#  TEMPERATURE ANNEALING (from NNM-KD v3)
# ============================================================================
def get_temperature(step, total_steps, T_max=3.0, T_min=1.0):
    progress = min(1.0, step / max(1, total_steps))
    return T_min + 0.5 * (T_max - T_min) * (1 + math.cos(math.pi * progress))
 
 
# ============================================================================
#  DIFFICULTY-AWARE WEIGHTING (from NNM-KD v3)
# ============================================================================
@torch.no_grad()
def compute_difficulty_weights(model, input_ids, attention_mask, early_layer_idx=2):
    """JSD(early layer, final layer) per token -> weight [0.5, 3.0]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True, use_cache=False)
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is None:
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        return None
    h_early = out.hidden_states[early_layer_idx]
    logits_early = lm_head(h_early)
    logits_final = out.logits
    p_early = F.softmax(logits_early.float(), dim=-1)
    p_final = F.softmax(logits_final.float(), dim=-1)
    m = 0.5 * (p_early + p_final)
    kl_early = (p_early * (p_early.log() - m.log()).clamp(min=-100)).sum(-1)
    kl_final = (p_final * (p_final.log() - m.log()).clamp(min=-100)).sum(-1)
    jsd = 0.5 * (kl_early + kl_final)
    jsd_norm = jsd / (jsd.mean() + 1e-8)
    return jsd_norm.clamp(0.5, 3.0)
 
 
# ============================================================================
#  TOKEN-LEVEL FILTERING (STAPO + Beyond 80/20 + Top-K) — from NNM-KD v3
# ============================================================================
def apply_token_filters(s_logits_active, t_logits_active,
                        s2t_tau_p=None, s2t_tau_h=None,
                        high_ent_rho=None, top_k=None, T=1.0):
    """
    Apply all v3 token-level filters to active logits:
      1. S2T spurious mask (STAPO)
      2. High-entropy selection (Beyond 80/20)
      3. Top-K logit filtering
      4. Temperature scaling
    Returns filtered (s, t) ready for KL computation.
    """
    s = s_logits_active.float()
    t = t_logits_active.float()
    if s.shape[0] == 0:
        return s, t
 
    # 1. S2T spurious token mask
    if s2t_tau_p is not None and s2t_tau_h is not None:
        with torch.no_grad():
            s_probs = F.softmax(s, dim=-1)
            s_entropy = -(s_probs * s_probs.log().clamp(min=-100)).sum(-1)
            t_top = t.argmax(dim=-1)
            s_prob_for_t_top = s_probs.gather(1, t_top.unsqueeze(1)).squeeze(1)
            spurious = (s_prob_for_t_top < s2t_tau_p) & (s_entropy < s2t_tau_h)
            valid = ~spurious
        if valid.any() and valid.sum() < s.shape[0]:
            s = s[valid]
            t = t[valid]
 
    # 2. High-entropy selection
    if high_ent_rho is not None and high_ent_rho < 1.0 and s.shape[0] > 4:
        with torch.no_grad():
            s_probs_ent = F.softmax(s, dim=-1)
            s_ent = -(s_probs_ent * s_probs_ent.log().clamp(min=-100)).sum(-1)
            k = max(1, int(high_ent_rho * s_ent.shape[0]))
            _, topk_idx = s_ent.topk(k)
        s = s[topk_idx]
        t = t[topk_idx]
 
    # 3. Top-K logit filtering
    vocab = s.shape[-1]
    if top_k is not None and top_k < vocab:
        with torch.no_grad():
            t_topk_vals, t_topk_idx = t.topk(top_k, dim=-1)
        s = s.gather(-1, t_topk_idx) / T
        t = t_topk_vals / T
    else:
        s = s / T
        t = t / T
 
    return s, t
 
 
# ============================================================================
#  CENTROID PRE-PASS (from NNM-KD)
# ============================================================================
@torch.no_grad()
def build_teacher_centroids(teacher, projector, dataloader,
                            t_mid, s_mid, device_t, device_s,
                            K, d_s, eta, T_dead, max_batches=3000):
    centroids = {s_lid: RunningCentroids(K, d_s, eta, T_dead, device_s)
                 for s_lid in s_mid}
    teacher.eval(); projector.eval()
    for i, (model_batch, no_model_batch, gen_data, _, _) in enumerate(
            tqdm(dataloader, desc="  Teacher centroid pre-pass", total=max_batches)):
        if i >= max_batches:
            break
        ids = model_batch["input_ids"].to(device_t)
        mask = model_batch["attention_mask"].to(device_t)
        t_act, _, _ = forward_with_hiddens(teacher, ids, mask, t_mid, device_t, no_grad=True)
        for t_lid, s_lid in zip(t_mid, s_mid):
            h_proj = projector(t_act[t_lid].to(device_s)).float()
            centroids[s_lid].update(h_proj)
    projector.train()
    return centroids
 
 
# ============================================================================
#  ORIGINAL DistiLLM FUNCTIONS (kept intact)
# ============================================================================
def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path, config=config,
                device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path, config=config,
                device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
 
        if args.teacher_peft_path is not None:
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
 
        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)
 
    model.eval()
    return model
 
 
def get_optimizer(args, model):
    while isinstance(model, DDP):
        model = model.module
    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer
 
 
def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters, power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")
    return lr_scheduler
 
 
def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    model = get_model(args, device)
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, args=args,
        lr_scheduler=lr_scheduler, mpu=None, config_params=ds_config)
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler
 
 
def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train",
                                        args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid",
                                      args.dev_num, args.dev_ratio, rng_sample)
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train",
                                           args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data
 
 
def pt_loss(args, model, model_batch, no_model_batch):
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
 
 
def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
 
    if args.model_parallel:
        raise NotImplementedError
 
    # ── NNM v3: apply token-level filters before KL ──
    if getattr(args, 'use_nnm_filters', False):
        # Extract active tokens
        labels = no_model_batch["label"]
        shift_s = logits[..., :-1, :]; shift_t = teacher_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        active = (shift_labels != -100)
        vocab = min(shift_s.size(-1), shift_t.size(-1))
        s_active = shift_s[active][..., :vocab]
        t_active = shift_t[active][..., :vocab]
 
        T_cur = getattr(args, '_current_temperature', 1.0)
 
        s_filt, t_filt = apply_token_filters(
            s_active, t_active,
            s2t_tau_p=NNM_CFG["s2t_tau_p"],
            s2t_tau_h=NNM_CFG["s2t_tau_h"],
            high_ent_rho=NNM_CFG["high_ent_rho"],
            top_k=NNM_CFG["top_k_logits"],
            T=T_cur,
        )
        # Use skewed forward KL on filtered tokens
        fake_labels = torch.zeros(s_filt.shape[0], dtype=torch.long, device=s_filt.device)
        mask = (fake_labels != -100).float()
 
        if "sfkl" in args.type:
            t_probs = F.softmax(t_filt, dim=-1); s_probs = F.softmax(s_filt, dim=-1)
            mixed = args.skew_alpha * t_probs + (1 - args.skew_alpha) * s_probs
            m_lp = torch.log(mixed.clamp(min=1e-10))
            inf_mask = torch.isinf(s_filt) | torch.isinf(t_filt)
            prod = torch.masked_fill(t_probs * m_lp, inf_mask, 0.).sum(-1)
            distil_loss = -(prod * mask).sum() / mask.sum().clamp(min=1) * (T_cur ** 2)
        elif "fkl" in args.type or args.type == "kd":
            t_probs = F.softmax(t_filt, dim=-1); s_lp = F.log_softmax(s_filt, dim=-1)
            inf_mask = torch.isinf(s_filt) | torch.isinf(t_filt)
            prod = torch.masked_fill(t_probs * s_lp, inf_mask, 0.).sum(-1)
            distil_loss = -(prod * mask).sum() / mask.sum().clamp(min=1) * (T_cur ** 2)
        elif "rkl" in args.type:
            s_probs = F.softmax(s_filt, dim=-1)
            s_lp = F.log_softmax(s_filt, dim=-1); t_lp = F.log_softmax(t_filt, dim=-1)
            inf_mask = torch.isinf(s_filt) | torch.isinf(t_filt)
            prod = torch.masked_fill(s_probs * (t_lp - s_lp), inf_mask, 0.).sum(-1)
            distil_loss = -(prod * mask).sum() / mask.sum().clamp(min=1) * (T_cur ** 2)
        else:
            # Fallback: use original DistiLLM functions without filtering
            distil_loss = _get_distil_loss_original(args, logits, teacher_logits, no_model_batch)
        return distil_loss
 
    return _get_distil_loss_original(args, logits, teacher_logits, no_model_batch)
 
 
def _get_distil_loss_original(args, logits, teacher_logits, no_model_batch):
    """Original DistiLLM KL loss dispatch (no NNM filters)."""
    if "sfkl" in args.type:
        return skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif "srkl" in args.type:
        return skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    elif "jsd" in args.type:
        return js_distance(logits, teacher_logits, no_model_batch)
    elif "tvd" in args.type:
        return tv_distance(logits, teacher_logits, no_model_batch)
    elif "fkl" in args.type or args.type == "kd":
        return forward_kl(logits, teacher_logits, no_model_batch)
    elif "rkl" in args.type:
        return reverse_kl(logits, teacher_logits, no_model_batch)
    elif "csd" in args.type:
        return csd(logits, teacher_logits, no_model_batch)
    else:
        raise NotImplementedError
 
 
def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, max_length=args.max_length,
            top_k=0, top_p=1, temperature=1.0, do_sample=True,
            return_dict_in_generate=True, output_scores=False)
    full_ids = t_gen_out.sequences
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]
    labels = torch.masked_fill(labels, mask == 0, -100)
    labels[:, :model_batch["input_ids"].size(1) - 1] = -100
    new_batch = {"input_ids": input_ids, "attention_mask": mask}
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask == 0, 0)
        new_batch["position_ids"] = position_ids
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    outputs = model(**new_batch, return_dict=True, use_cache=False)
    return loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
 
 
# ============================================================================
#  MAIN TRAINING LOOP — DistiLLM + NNM-KD features
# ============================================================================
def finetune(args, tokenizer, model, optimizer, lr_scheduler,
             dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning (DistiLLM + NNM-KD v3)")
 
    if args.model_parallel:
        raise NotImplementedError
 
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()
 
    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True,
                                  rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=dataset["train"].collate)
 
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True,
                                         rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
            dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size,
            num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
 
    student_generator = SampleGenerator(args, tokenizer)
 
    # ── NNM SETUP ────────────────────────────────────────────────
    use_nnm = getattr(args, 'use_nnm', False) and teacher_model is not None
    projector = None
    R = None
    t_cents, s_cents = {}, {}
    t_mid, s_mid = [], []
    lw_dict = {}
 
    if use_nnm:
        print_rank("Setting up NNM components...")
        # Get hidden dims from student model
        student_module = model.module if hasattr(model, 'module') else model
        teacher_module = teacher_model
 
        d_t = teacher_module.config.hidden_size
        d_s = student_module.config.hidden_size
        L_t = teacher_module.config.num_hidden_layers
        L_s = student_module.config.num_hidden_layers
 
        t_mid = select_mid_layers(L_t, NNM_CFG["n_mid_layers"])
        s_mid = select_mid_layers(L_s, NNM_CFG["n_mid_layers"])
        lw_dict = {s_lid: layer_weight(s_lid, L_s, NNM_CFG["sigma_layer"]) for s_lid in s_mid}
 
        print_rank(f"  Teacher hidden={d_t}, layers={L_t}, mid={t_mid}")
        print_rank(f"  Student hidden={d_s}, layers={L_s}, mid={s_mid}")
 
        projector = HiddenProjector(d_t, d_s).to(device)
        R = make_R(d_s, NNM_CFG["d_prime"], device)
 
        # Centroid pre-pass
        print_rank("  Building teacher centroids...")
        t_cents = build_teacher_centroids(
            teacher_model, projector, train_dataloader,
            t_mid, s_mid, device, device,
            NNM_CFG["K_centroids"], d_s,
            NNM_CFG["eta_centroid"], NNM_CFG["T_dead"],
            max_batches=NNM_CFG["centroid_prepass_batches"],
        )
        s_cents = {
            s_lid: RunningCentroids(NNM_CFG["K_centroids"], d_s,
                                     NNM_CFG["eta_centroid"], NNM_CFG["T_dead"], device)
            for s_lid in s_mid
        }
 
        # Add projector params to optimizer
        for pg in optimizer.param_groups:
            pg['params'].extend(list(projector.parameters()))
 
    # Enable NNM filters in distil loss
    args.use_nnm_filters = getattr(args, 'use_nnm_filters', use_nnm)
 
    step, global_step = 1, 1
    total_loss, total_distil_loss, total_nnm_loss, total_time = 0.0, 0.0, 0.0, 0.0
 
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    replay_buffer = ReplayBuffer(args)
 
    def nnm_weight_fn(s):
        return NNM_CFG["lambda_nnm"] * min(1.0, s / max(1, NNM_CFG["nnm_warmup"]))
 
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
 
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
 
            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
 
            torch.cuda.synchronize()
            st_time = time.time()
 
            # ── Temperature annealing (NNM v3) ──
            T_cur = get_temperature(global_step, args.total_iters,
                                     NNM_CFG["kl_temp_max"], NNM_CFG["kl_temp_min"])
            args._current_temperature = T_cur
 
            # ── Adaptive sampling (DistiLLM) ──
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
 
            # ── Student generation (DistiLLM) ──
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(
                        model_batch, no_model_batch, gen_data, device)
                elif "adaptive" in args.type and (
                        r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids')
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(
                        model_batch, no_model_batch, gen_data, device)
                model.train()
 
            # ── Forward pass ──
            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits
 
            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]),
                                     no_model_batch["label"].view(-1))
 
            # ── Distillation loss (with NNM token filters if enabled) ──
            if teacher_model is not None:
                distil_loss = get_distil_loss(args, tokenizer, model, teacher_model,
                                              model_batch, no_model_batch, logits)
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                loss = lm_loss
                distil_loss = torch.tensor(0.)
 
            # ── NNM LOSS (from NNM-KD) ──
            nnm_loss_val = torch.tensor(0., device=device)
            if use_nnm:
                student_module = model.module if hasattr(model, 'module') else model
                labels = no_model_batch["label"]
                label_mask = (labels != -100)
 
                # Student forward with hiddens
                s_act, s_full, _ = forward_with_hiddens(
                    student_module, model_batch["input_ids"],
                    model_batch["attention_mask"],
                    s_mid, device, no_grad=False, label_mask=label_mask)
 
                # Teacher forward with hiddens (no grad)
                t_act, t_full, _ = forward_with_hiddens(
                    teacher_model, model_batch["input_ids"],
                    model_batch["attention_mask"],
                    t_mid, device, no_grad=True)
 
                # Project + correct teacher hiddens
                lam_nnm = nnm_weight_fn(global_step)
                label_mask_float = label_mask.float()
 
                for t_lid, s_lid in zip(t_mid, s_mid):
                    h_t = t_full[t_lid].to(device)
                    B, T_len, _ = h_t.shape
                    h_proj = projector(h_t.reshape(-1, d_t)).detach()
 
                    if NNM_CFG["do_teacher_correction"]:
                        flat_mask = model_batch["attention_mask"].reshape(-1).bool()
                        h_active = h_proj[flat_mask]
                        h_corr = correct_teacher_hiddens(
                            h_active, t_cents[s_lid].C, R,
                            NNM_CFG["tc_lambda"], NNM_CFG["ns_iters"], NNM_CFG["tc_steps"])
                        h_corr_full = h_proj.clone()
                        h_corr_full[flat_mask] = h_corr.to(h_proj.dtype)
                        h_t_proj = h_corr_full.reshape(B, T_len, d_s)
                    else:
                        h_t_proj = h_proj.reshape(B, T_len, d_s)
 
                    label_flat = label_mask_float.reshape(-1).bool()
                    h_t_active = h_t_proj.reshape(-1, d_s)[label_flat]
                    h_s_active = s_act[s_lid]
 
                    nnm_loss_val = nnm_loss_val + nnm_loss_one_layer(
                        h_s_active, h_t_active,
                        s_cents[s_lid].C, t_cents[s_lid].C,
                        R, lw_dict[s_lid], NNM_CFG["ns_iters"])
 
                nnm_loss_val = nnm_loss_val / len(s_mid)
                loss = loss + lam_nnm * nnm_loss_val
 
                # Update student centroids
                with torch.no_grad():
                    for s_lid in s_mid:
                        s_cents[s_lid].update(s_act[s_lid].float().detach())
 
            # ── Pre-trained data mixing (DistiLLM) ──
            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)
 
            model.backward(loss)
            model.step()
 
            # ── Logging ──
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size
 
            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
 
            total_nnm_loss += nnm_loss_val.item()
 
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time
            total_loss += global_loss
            total_time += elapsed_time
 
            def get_log(log_loss, log_distil_loss, log_time):
                nnm_str = f" | nnm: {total_nnm_loss / max(step, 1):.4f}" if use_nnm else ""
                return ("train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
                        "loss: {:.4f} | ds_loss: {:.4f}{} | T: {:.2f} | lr: {:.4e} | "
                        "time: {:.3f}").format(
                    epoch, step, args.total_iters * args.gradient_accumulation_steps,
                    global_step, args.total_iters, log_loss, log_distil_loss, nnm_str,
                    T_cur, lr_scheduler.get_last_lr()[0], log_time)
 
            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))
 
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / args.log_interval)
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_nnm_loss, total_time = 0.0, 0.0, 0.0, 0.0
 
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if not args.model_parallel:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                        if projector is not None:
                            torch.save(projector.state_dict(),
                                       os.path.join(save_dir_path, "projector.pt"))
                dist.barrier()
 
            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev",
                                          epoch, device, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss
                model.train()
 
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            if global_step > args.total_iters:
                break
 
    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=[tokenizer.eos_token_id, 151643],
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
        teacher_model.resize_token_embeddings(model.module.config.vocab_size)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()