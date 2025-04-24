# DBPO — Direction-Based Policy Optimization  
*A proof-of-concept you can copy cell-by-cell into a Jupyter / Colab notebook.*

---

> We keep the **exact same Python script**, only broken into smaller blocks and sprinkled with short comments that point back to the maths.  
> Copy each block in order and run it — the behaviour is identical to the single-file version.

---

## 0 · Warm-up talk

DBPO steers a GPT-2 policy by **nudging its hidden-state vectors** toward directions that correlate with a scalar reward (here : the number of “happy” words).  
A frozen reference model supplies a KL penalty so the policy does not drift too far.

---

## 1 · Imports, hyper-params & speed flags

The first thing we need is a bit of boiler-plate to configure the run.

```python
#!/usr/bin/env python3
"""
DBPO — Direction-Based Policy Optimisation on GPT-2-small  
(math: we will later minimise  L = L_dir + β L_KL )
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange

# ---------------------- Config & Flags ----------------------------
MODEL_NAME       = "gpt2"
BATCH_SIZE       = 30
PROMPT           = "Hello i am so"
STEPS            = 150
GEN_MAX_TOK      = 80
HIDDEN_LAYER     = -3          # use layer −3 for hidden states
LAMBDA_RIDGE     = 0.08        # λ in ridge regression
ALPHA_DIR        = 10.0        # α in  L_dir
KL_BETA          = 0.02        # β in  L_KL
LR               = 2e-5
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
EPS              = 2e-3

# Speed-up flags
USE_FLASH_ATTENTION = True
USE_KV_CACHE        = True
USE_QUANT           = False
USE_PARALLEL_REWARD = True
USE_CHOLESKY_SOLVE  = True
USE_CUDA_PIPELINE   = True
USE_DIM_REDUCTION   = False
PROJ_DIM            = 768
```

---

> **What’s going on?**  
> *Nothing fancy yet dw*

---

## 2 · The reward and the models

We define a tiny lexical reward and then load a trainable policy plus a frozen reference copy.

```python
# ---------------------- Reward Function ---------------------------
def happy_reward(text: str) -> float:
    happy_keywords = [
        "happy", "joy", "smile", "love", "great", "wonderful", "fantastic",
        "excited", "fun", "awesome", "delight", "pleasure", "laugh", "amazing",
        "beautiful", "cheerful", "glad", "sunshine", "sweet", "positive",
        "grateful", "thankful", "satisfied", "content", "enjoy", "yay", "woohoo",
        "celebrate", "blessed", "bright", "good", "incredible", "peaceful", 
        "relaxed", "thrilled", "elated", "jolly", "bliss", "ecstatic", 
        "upbeat", "serene", "radiant", "lively", "vibrant", "charming", 
        "heartwarming", "smiling", "giggling", "beaming", "sparkling", 
        "merry", "bubbly", "harmonious", "euphoric", "high-spirited", 
        "overjoyed", "rejoice", "sunny", "laughing", "zest", "contented",
        "positive vibes", "good vibes", "cheery", "elation"
    ]
    text_lower = text.lower()
    reward = 0.0
    for word in happy_keywords:
        if word in text_lower:
            reward += 1.0
    return reward

# -------------- Load tokenizer & models ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

config = AutoConfig.from_pretrained(MODEL_NAME)
config.use_flash_attention_2 = True  # toggle Flash-Attn v2 if present

# — trainable policy —
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, config=config, torch_dtype=torch.bfloat16
).to(DEVICE)
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.gradient_checkpointing_enable()

# — frozen reference (for KL) —
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, config=config, torch_dtype=torch.bfloat16
).to(DEVICE)
ref_model.config.pad_token_id = tokenizer.eos_token_id
ref_model.config.eos_token_id = tokenizer.eos_token_id
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Runtime helpers
executor   = ThreadPoolExecutor() if USE_PARALLEL_REWARD else None
upd_stream = torch.cuda.Stream() if (USE_CUDA_PIPELINE and DEVICE.startswith("cuda")) else None

print(f"\nLoaded {MODEL_NAME} on {DEVICE} with optimisations:")
print(f"  FlashAttn={USE_FLASH_ATTENTION}  KVcache={USE_KV_CACHE}  FP16={USE_QUANT}")
print(f"  ParallelReward={USE_PARALLEL_REWARD}  Cholesky={USE_CHOLESKY_SOLVE}")
print(f"  Pipeline={USE_CUDA_PIPELINE}  DimReduction={USE_DIM_REDUCTION}\n")
```

---

### Side note — the maths we will need later

*Pooling hidden states*

```math
\bar h_i \;=\; \frac{1}{L_i}\sum_{t=1}^{L_i} h_{i,t}
```

*Ridge regression to find the “good” direction*

```math
w^{\star} = \bigl(H^{\!\top}H + \lambda I\bigr)^{-1} H^{\!\top} r
```

*A mini-batch loss*

```math
\mathcal{L}
\;=\; -\alpha\,\frac{1}{B}\sum_{i=1}^B \bar h_i^{\!\top} w^{\star}
      +\beta\,\text{KL}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr)
```

> *If this looks complicated: we push hidden states toward the reward-direction and pay a KL tax.*

---

## 3 · One optimisation step

Below is the heart of DBPO. New inline comments reference the equations above.

```python
# --- Revised rl_step helper ---
def rl_step(prompts):
    # 1) Sample continuations (no gradient)
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        seqs = model.generate(
            **enc,
            max_new_tokens=GEN_MAX_TOK,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            use_cache=USE_KV_CACHE,
            return_dict_in_generate=False
        )
    completions = tokenizer.batch_decode(
        seqs[:, enc.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # 2) Forward pass to get hidden states & logits
    full_attn = torch.ones_like(seqs, device=DEVICE)
    outputs = model(
        input_ids=seqs,
        attention_mask=full_attn,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True
    )
    hs          = outputs.hidden_states[HIDDEN_LAYER]   # [B, seq, d]
    main_logits = outputs.logits                       # [B, seq, V]

    # 2b) Reference logits (for KL)
    with torch.no_grad():
        ref_outs = ref_model(
            input_ids=seqs,
            attention_mask=full_attn,
            use_cache=False,
            return_dict=True
        )
    ref_logits = ref_outs.logits                       # [B, seq, V]

    # 3) Pool hidden states  (math:  \bar h_i )
    mask    = full_attn.unsqueeze(-1)
    h_mean  = (hs * mask).sum(dim=1) / mask.sum(dim=1)  # [B, d]

    # 4) Rewards
    rewards_list = (list(executor.map(happy_reward, completions))
                    if USE_PARALLEL_REWARD
                    else [happy_reward(c) for c in completions])
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=DEVICE)

    # 5) Ridge regression  (math: solve for  w^{\star})
    H_fp32 = h_mean.to(torch.float32).detach()
    r_fp32 = rewards.unsqueeze(1)
    d = H_fp32.shape[1]
    I_d = torch.eye(d, device=DEVICE, dtype=torch.float32)
    w_hat = torch.linalg.solve(H_fp32.T @ H_fp32 + LAMBDA_RIDGE * I_d,
                               H_fp32.T @ r_fp32).squeeze()

    # 6) KL term (approx token-wise)
    main_lps = F.log_softmax(main_logits[:, :-1, :], dim=-1)
    ref_lps  = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
    tokens   = seqs[:, 1:]
    kl_loss = (main_lps.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
               - ref_lps.gather(2, tokens.unsqueeze(-1)).squeeze(-1)).mean()

    # 7) Total loss  L_dir + β L_KL
    optimizer.zero_grad()
    dir_loss   = -ALPHA_DIR * (h_mean.to(torch.float32) @ w_hat).mean()
    total_loss = dir_loss + KL_BETA * kl_loss
    total_loss.backward()
    optimizer.step()

    # 8) House-keeping to avoid GPU memory leaks
    del outputs, ref_outs, H_fp32, r_fp32
    gc.collect()
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "raw_rewards":      rewards_list,
        "w_hat_norm":       w_hat.norm().item(),
        "directional_loss": dir_loss.item(),
        "kl_loss":          kl_loss.item()
    }, completions[0]
```

---

> **Why so many deletes?**  
> Because we really wanna get rid of the excess memory building up.

---

## 4 · Training loop

Time to iterate.

```python
# --- Training Loop ---
summed_raw_rewards = []
for step in trange(1, STEPS + 1, desc="DBPO-steps"):
    batch_prompts = [PROMPT] * BATCH_SIZE
    metrics, sample = rl_step(batch_prompts)

    # Logging
    sum_raw = sum(metrics["raw_rewards"])
    summed_raw_rewards.append(sum_raw)
    print(f"\nStep {step:02d}/{STEPS}")
    print(f"  Σ raw rewards : {sum_raw:.4f}")
    print(f"  rewards       : {metrics['raw_rewards']}")
    print(f"  ||w_hat||₂    : {metrics['w_hat_norm']:.2f}")
    print(f"  dir loss      : {metrics['directional_loss']:.4f}")
    print(f"  KL loss       : {metrics['kl_loss']:.4f}")
    print(f"  sample        : \"{sample}\"\n")

print("Training done ✅")
```

---

## 5 · And that’s it

You now have DBPO split into digestible chunks:

1. **Config** & imports  
2. Reward + model loading  
3. A mathematically-annotated optimisation step  
4. The outer training loop  

Feel free to drop the blocks into separate notebook cells, play with the hyper-parameters, or swap the reward function for something else.
