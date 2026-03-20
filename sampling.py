import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature):
    """
    Sample from the Gumbel-Softmax distribution and optionally apply softmax.
    """
    if temperature == 0.0:
        return logits
    else:
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        return logits / temperature + gumbel_noise

@torch.no_grad()
def arm_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False):
    """
    Autoregressive sampling:
      - generates one token at a time, left-to-right
      - starts at the first mask_id position (prompt end)
      - uses gumbel-softmax + argmax (same style as mdm_sampling)
      - if eos_id is provided, stops per sequence at first EOS and fills the rest with EOS
    """
    temperature = sampling_cfg.temperature
    eos_id = getattr(sampling_cfg, "eos_id", None)

    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = []

    # start position per row = first mask_id (prompt end)
    is_mask = (xt == mask_id)
    any_mask = is_mask.any(dim=1)
    start_pos = is_mask.float().argmax(dim=1)                 # 0 if no mask
    start_pos = torch.where(any_mask, start_pos, torch.full_like(start_pos, L))

    # nothing to generate
    if int(start_pos.min().item()) >= L:
        # if eos_id is set, still ensure "fill to L with eos" is trivially satisfied (nothing to do)
        return (xt, track_xt) if track else xt

    # track which sequences are finished (EOS generated)
    done = torch.zeros(B, dtype=torch.bool, device=xt.device) if eos_id is not None else None

    for pos in range(L):
        # only generate for rows where:
        #  - pos is in the generation region
        #  - token is still mask
        #  - and (if eos_id) we haven't already produced EOS
        can_fill = (pos >= start_pos) & (xt[:, pos] == mask_id)
        if eos_id is not None:
            can_fill = can_fill & (~done)

        if not can_fill.any():
            continue

        # to predict token at `pos`, we use logits at `pos-1`
        src = pos - 1
        if src < 0:
            continue

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(xt)  # (B, L, V)

        logits_step = logits[:, src, :]  # predicts token at pos
        noisy = gumbel_softmax(logits_step, temperature=temperature)  # (B, V)
        next_tok = torch.argmax(noisy, dim=-1)                        # (B,)

        xt[can_fill, pos] = next_tok[can_fill]

        if eos_id is not None:
            # mark sequences as done if EOS was just produced at this position
            done = done | (can_fill & (next_tok == eos_id))

        if track:
            track_xt.append(xt.clone().detach().cpu())

    # If eos_id is set: for any sequence that has EOS in the generation region,
    # fill all later positions (up to length L) with EOS.
    if eos_id is not None:
        pos_idx = torch.arange(L, device=xt.device).unsqueeze(0)  # (1, L)
        gen_region = pos_idx >= start_pos.unsqueeze(1)            # (B, L)
        eos_in_region = (xt == eos_id) & gen_region               # (B, L)
        any_eos = eos_in_region.any(dim=1)                        # (B,)

        if any_eos.any():
            first_eos = eos_in_region.float().argmax(dim=1)       # (B,) (meaningful only where any_eos)
            # fill strictly after first_eos (and within gen_region) with eos_id
            fill_mask = any_eos.unsqueeze(1) & gen_region & (pos_idx > first_eos.unsqueeze(1))
            xt[fill_mask] = eos_id

            if track:
                # optional: record the post-fill final state once
                track_xt.append(xt.clone().detach().cpu())

    return (xt, track_xt) if track else xt

@torch.no_grad()
def mdm_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False, arm_init: bool = False, prompt_mask: torch.Tensor = None):
    # sampling hyperparameters
    # xt can include clean tokens
    # if track == True, we return the trace (used for the debugging purpose)

    # To be compatible with multi-step proseco
    strategy = sampling_cfg.get("strategy", "default")
    if strategy in ["multi_proseco", "multi_proseco_buggy"]:
        return mdm_multi_proseco_sampling(model, xt, mask_id, sampling_cfg, device, track, arm_init, prompt_mask)
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    edit_freq = sampling_cfg.get("edit_freq", -1) # omega
    edit_step = sampling_cfg.get("edit_step", 0) # S
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool) # All responses
    if edit_freq > 0:
        assert edit_step >=0, "edit_step must be non-negative when edit_freq is set"

    # shape
    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = []

    if arm_init:
        xt_t1, xt = xt[:, :1], xt[:, 1:]
        L = L - 1

    for i in range(L // unmasking_num + 1):
        # mask indicies
        mask_indices = (xt == mask_id)

        if mask_indices.sum() == 0:
            break

        # calculate logits
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
            logits = model(torch.cat([xt_t1, xt], dim=1) if arm_init else xt) # [B, L, V]

        if edit_freq > 0 and (i + 1) % edit_freq == 0:
            xt, logits = mdm_edit_sampling(logits, model, xt, mask_id, sampling_cfg, prompt_mask, device, arm_init)

        if arm_init:
            logits = logits[:, :-1, :]
        logits_with_noise = gumbel_softmax(logits, temperature = temperature)
        p = F.softmax(logits, dim = -1)

        if confidence == "top_k":
            unmasking_score = torch.where(mask_indices, p.max(dim = -1).values, -float('inf'))
        elif confidence == "top_k_margin":
            probs_top_2 = p.topk(k=2, dim=-1).values
            unmasking_score = torch.where(mask_indices, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
        elif confidence == "entropy":
            entropy = (- p * torch.log(p + 1e-10)).sum(dim = -1)
            unmasking_score = torch.where(mask_indices, entropy, -float('inf'))
        elif confidence == "random":
            raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
        else:
            raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
        
        # update masked tokens by selecting top-k per batch this step
        k_max = min(unmasking_num, int(mask_indices.sum(dim=-1).max().item()))
        
        if k_max > 0:
            # 2. Get top-k indices for the entire batch at once
            _, select_indices = torch.topk(unmasking_score, k=k_max, dim=-1) # Shape: [B, k_max]
            
            # 3. Filter invalid selections. 
            # If a sequence has fewer than k_max masked tokens, topk will grab some 
            # already-unmasked tokens (score -inf). We identify valid ones using the original mask.
            valid_selections = mask_indices.gather(1, select_indices) # Shape: [B, k_max]
            
            # 4. Create a full [B, L] boolean mask for where updates should occur
            update_mask = torch.zeros_like(mask_indices)
            update_mask.scatter_(1, select_indices, valid_selections)
            
            # 5. Compute argmax for the whole tensor and apply updates conditionally
            new_tokens = torch.argmax(logits_with_noise, dim=-1) # Shape: [B, L]
            xt = torch.where(update_mask, new_tokens, xt)

        if track:
            cur = torch.cat([xt_t1, xt], dim=1) if arm_init else xt
            track_xt.append(cur.clone().detach().cpu())
    if arm_init:
        xt = torch.cat([xt_t1, xt], dim=1)
    if track:
        track_xt = torch.stack(track_xt, dim=0) # (T, B, L)
        return xt, track_xt
    else:
        return xt

@torch.no_grad()    
def mdm_edit_sampling(logits, model, xt, mask_id, sampling_cfg, prompt_mask, device: torch.device = None, arm_init: bool = False):
    '''
    Learning Unmasking Policies for Diffusion Language Models (http://arxiv.org/abs/2512.09106)
    '''
    # self correction sampling. Currently only implements greedy sampling
    assert arm_init == False, "ARM initialization is not compatible with edit sampling currently"

    B, L = xt.shape
    edit_step = sampling_cfg.edit_step
    unmask_indices = (xt != mask_id)
    if unmask_indices.sum() == 0:
        return xt, logits
    
    new_pred = torch.argmax(logits, dim=-1) # [B, L]
    yt = torch.where(~prompt_mask, new_pred, xt) # only edit the response part, keep the prompt unchanged
    for i in range(edit_step):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
            logits = model(yt) # [B, L, V]
        new_pred = torch.argmax(logits, dim=-1) # [B, L]
        yt = torch.where(~prompt_mask, new_pred, xt) # only update the response part
    # Replace unmasked tokens
    xt = torch.where(unmask_indices, yt, xt)
    return xt, logits
    
def mdm_multi_proseco_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False, arm_init: bool = False, prompt_mask: torch.Tensor = None):
    '''
    A new evaluation method for our multi step proseco.
    For each forward pass, replace all clean tokens, and reveal part of the masks
    '''
    # sampling hyperparameters
    # xt can include clean tokens
    # if track == True, we return the trace (used for the debugging purpose)
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    correction_step = sampling_cfg.get("correction_step", 0) # number of self-correction steps after revealing all tokens
    strategy = sampling_cfg.get("strategy", "default")
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(xt, dtype=torch.bool) # All responses
    eff_L = (~prompt_mask).sum(dim=1).max().item()

    # shape
    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = [xt.clone().detach().cpu()]

    if arm_init:
        xt_t1, xt = xt[:, :1], xt[:, 1:]
        L = L - 1
        eff_L = eff_L - 1

    # Warning: L is prompt + response length, not just response length
    for i in range((eff_L-1) // unmasking_num + 1 + correction_step):
        # mask indicies
        mask_indices = (xt == mask_id)

        # if mask_indices.sum() == 0:
        #     break

        # calculate logits
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
            logits = model(torch.cat([xt_t1, xt], dim=1) if arm_init else xt) # [B, L, V]

        if arm_init:
            logits = logits[:, :-1, :]
        logits_with_noise = gumbel_softmax(logits, temperature = temperature)
        p = F.softmax(logits, dim = -1)

        if confidence == "top_k":
            unmasking_score = torch.where(mask_indices, p.max(dim = -1).values, -float('inf'))
        elif confidence == "top_k_margin":
            probs_top_2 = p.topk(k=2, dim=-1).values
            unmasking_score = torch.where(mask_indices, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
        elif confidence == "entropy":
            entropy = (- p * torch.log(p + 1e-10)).sum(dim = -1)
            unmasking_score = torch.where(mask_indices, entropy, -float('inf'))
        elif confidence == "random":
            raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
        else:
            raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
        
        # update masked tokens by selecting top-k per batch this step
        # 1. Find the maximum number of tokens to unmask across the batch
        k_max = min(unmasking_num, int(mask_indices.sum(dim=-1).max().item()))
        
        update_mask = torch.zeros_like(mask_indices)
        if k_max > 0:
            # 2. Get top-k indices for the entire batch at once
            _, select_indices = torch.topk(unmasking_score, k=k_max, dim=-1) # Shape: [B, k_max]
            
            # 3. Filter invalid selections. 
            # If a sequence has fewer than k_max masked tokens, topk will grab some 
            # already-unmasked tokens (score -inf). We identify valid ones using the original mask.
            valid_selections = mask_indices.gather(1, select_indices) # Shape: [B, k_max]
            
            # 4. Create a full [B, L] boolean mask for where updates should occur
            update_mask.scatter_(1, select_indices, valid_selections)
            
            # 5. Compute argmax for the whole tensor and apply updates conditionally
            new_tokens = torch.argmax(logits_with_noise, dim=-1) # Shape: [B, L]

        # update all revealed tokens, and part of the masks
        if strategy == "multi_proseco":
            reveal_indices = (~mask_indices) & (~prompt_mask) # only reveal the response part, keep the prompt unchanged
        elif strategy == "multi_proseco_buggy": # "multi_proseco_buggy"
            reveal_indices = prompt_mask | (~mask_indices) # update prompts and all the revealed tokens.
        else:
            raise NotImplementedError(f"Sampling strategy {strategy} is not supported. Expected 'multi_proseco' or 'multi_proseco_buggy'.")
        xt = torch.where(update_mask | reveal_indices, new_tokens, xt)

        if track:
            cur = torch.cat([xt_t1, xt], dim=1) if arm_init else xt
            track_xt.append(cur.clone().detach().cpu())
    if arm_init:
        xt = torch.cat([xt_t1, xt], dim=1)
    if track:
        track_xt = torch.stack(track_xt, dim=0) # (T, B, L)
        print(track_xt.shape)
        return xt, track_xt
    else:
        return xt

# def mdm_multi_proseco_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False, arm_init: bool = False, prompt_mask: torch.Tensor = None):
#     '''
#     A new evaluation method for our multi step proseco.
#     For each forward pass, replace all clean tokens, and reveal part of the masks
#     '''
#     # sampling hyperparameters
#     # xt can include clean tokens
#     # if track == True, we return the trace (used for the debugging purpose)
#     temperature = sampling_cfg.temperature
#     confidence = sampling_cfg.confidence
#     unmasking_num = sampling_cfg.unmasking_num
#     correction_step = sampling_cfg.get("correction_step", 0) # number of self-correction steps after revealing all tokens
#     if prompt_mask is None:
#         prompt_mask = torch.zeros_like(xt, dtype=torch.bool) # All responses

#     # shape
#     B, L = xt.shape
#     xt = xt.clone()
#     if track:
#         track_xt = []

#     if arm_init:
#         xt_t1, xt = xt[:, :1], xt[:, 1:]
#         L = L - 1

#     for i in range(2*((L-1) // unmasking_num + 1 + correction_step)):
#         # mask indicies
#         mask_indices = (xt == mask_id)

#         if mask_indices.sum() == 0:
#             break

#         # calculate logits
#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
#             logits = model(torch.cat([xt_t1, xt], dim=1) if arm_init else xt) # [B, L, V]

#         if arm_init:
#             logits = logits[:, :-1, :]
#         logits_with_noise = gumbel_softmax(logits, temperature = temperature)
#         p = F.softmax(logits, dim = -1)

#         if confidence == "top_k":
#             unmasking_score = torch.where(mask_indices, p.max(dim = -1).values, -float('inf'))
#         elif confidence == "top_k_margin":
#             probs_top_2 = p.topk(k=2, dim=-1).values
#             unmasking_score = torch.where(mask_indices, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
#         elif confidence == "entropy":
#             entropy = (- p * torch.log(p + 1e-10)).sum(dim = -1)
#             unmasking_score = torch.where(mask_indices, entropy, -float('inf'))
#         elif confidence == "random":
#             raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
#         else:
#             raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
        
#         # update masked tokens by selecting top-k per batch this step
#         # 1. Find the maximum number of tokens to unmask across the batch
#         if i % 2 == 0: # decoding step
#             k_max = min(unmasking_num, int(mask_indices.sum(dim=-1).max().item()))
            
#             update_mask = torch.zeros_like(mask_indices)
#             if k_max > 0:
#                 # 2. Get top-k indices for the entire batch at once
#                 _, select_indices = torch.topk(unmasking_score, k=k_max, dim=-1) # Shape: [B, k_max]
                
#                 # 3. Filter invalid selections. 
#                 # If a sequence has fewer than k_max masked tokens, topk will grab some 
#                 # already-unmasked tokens (score -inf). We identify valid ones using the original mask.
#                 valid_selections = mask_indices.gather(1, select_indices) # Shape: [B, k_max]
                
#                 # 4. Create a full [B, L] boolean mask for where updates should occur
#                 update_mask.scatter_(1, select_indices, valid_selections)
                
#                 # 5. Compute argmax for the whole tensor and apply updates conditionally
#                 new_tokens = torch.argmax(logits_with_noise, dim=-1) # Shape: [B, L]

#             # update all revealed tokens, and part of the masks
#             # xt = torch.where(update_mask | (~mask_indices), new_tokens, xt)
#             xt = torch.where(update_mask, new_tokens, xt)
#         else: # correction step, update all tokens
#             new_tokens = torch.argmax(logits_with_noise, dim=-1) # Shape: [B, L]
#             xt = torch.where(~prompt_mask, new_tokens, xt) # only update the response part, keep the prompt unchanged

#         if track:
#             cur = torch.cat([xt_t1, xt], dim=1) if arm_init else xt
#             track_xt.append(cur.clone().detach().cpu())
#     if arm_init:
#         xt = torch.cat([xt_t1, xt], dim=1)
#     if track:
#         track_xt = torch.stack(track_xt, dim=0) # (T, B, L)
#         return xt, track_xt
#     else:
#         return xt

@torch.no_grad()
def mdm_sampling_block(model, xt, block_size, mask_id, sampling_cfg, device: torch.device = None):
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    device = xt.device

    # shape 
    B, L = xt.shape
    assert L % block_size == 0, "block size must be divisible by the max_length"
    n_blocks = L // block_size
    xt = xt.clone()

    for n in range(n_blocks):
        s = n * block_size
        e = s + block_size

        for i in range(block_size // unmasking_num + 2):
            valid_mask_ids = (xt[:, s:e] == mask_id) # [B, e]
            if valid_mask_ids.sum() == 0:
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                logits = model(xt[:, : e]) # [B, e, V]

            logits_with_noise = gumbel_softmax(logits[:, s:e, :], temperature = temperature) # [B, e, V]
            p = F.softmax(logits[:, s:e, :], dim = -1)

            if confidence == "top_k":
                unmasking_score = torch.where(valid_mask_ids, p.max(dim = -1).values, -float('inf'))
            elif confidence == "top_k_margin":
                probs_top_2 = p.topk(k=2, dim=-1).values
                unmasking_score = torch.where(valid_mask_ids, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
            elif confidence == "random":
                raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
            else:
                raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
                
            for j in range(B):
                k = min(unmasking_num, int(valid_mask_ids[j].sum().item())) # number of tokens to unmask
                if k > 0:
                    _, select_indices = torch.topk(unmasking_score[j], k=k)
                    xt[j, s + select_indices] = torch.argmax(logits_with_noise[j, select_indices], dim = -1)
    
    assert (xt == mask_id).sum() == 0, "There are still masked tokens in the input"
            
    return xt