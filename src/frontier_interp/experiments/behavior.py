"""Behavioral distillation experiment.

This module supports two complementary behavioral comparisons:
- full-distribution distillation via KL on next-token logits,
- restricted-choice evaluation for datasets such as ARC, PIQA, and HellaSwag.
"""

from __future__ import annotations

import time
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from frontier_interp.analysis.metrics import behavior_distillation_metrics, restricted_choice_metrics
from frontier_interp.utils.time import format_seconds



def train_behavior_interpreter(interpreter, target_model, train_batches, steps: int, logger, run_label: str, lr: float, weight_decay: float):
    opt = torch.optim.AdamW(interpreter.parameters(), lr=lr, weight_decay=weight_decay)
    interpreter.train()

    baseline_loss = None
    start = time.time()
    batches = list(train_batches)

    for step in range(steps):
        batch = batches[step % len(batches)]
        token_batch = target_model.tokenize_batch([ex.text for ex in batch["examples"]], batch["max_prompt_len"])

        with torch.no_grad():
            target_logits, _ = target_model.extract_logits_and_attentions(token_batch, output_attentions=False)

        pred_logits = interpreter.forward_behavior(token_batch["input_ids"])
        loss = F.kl_div(
            F.log_softmax(pred_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction="batchmean",
        )

        if baseline_loss is None:
            uniform_logits = torch.zeros_like(pred_logits)
            baseline_loss = F.kl_div(
                F.log_softmax(uniform_logits, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="batchmean",
            ).item()
            baseline_loss = max(baseline_loss, 1e-8)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - start
            done = step + 1
            eta = (steps - done) * (elapsed / done)
            logger.log(
                f"{run_label}[behavior] step={step:4d}/{steps} "
                f"| loss={loss.item():.4f} "
                f"| normed={loss.item()/baseline_loss:.4f} "
                f"| elapsed={format_seconds(elapsed)} "
                f"| eta={format_seconds(eta)}"
            )

    return baseline_loss


@torch.no_grad()
def eval_behavior_interpreter(interpreter, target_model, val_batches, baseline_loss: float, max_prompt_len: int):
    interpreter.eval()
    rows = []
    example_artifacts = []

    for batch in val_batches:
        texts = [ex.text for ex in batch["examples"]]
        token_batch = target_model.tokenize_batch(texts, batch["max_prompt_len"])
        target_logits, _ = target_model.extract_logits_and_attentions(token_batch, output_attentions=False)
        pred_logits = interpreter.forward_behavior(token_batch["input_ids"])

        metrics = behavior_distillation_metrics(pred_logits, target_logits)
        metrics["normed_loss"] = metrics["kl_loss"] / baseline_loss
        metrics["family"] = batch["family"]
        metrics["source"] = batch["source"]

        # restricted-choice behavioral evaluation when available
        choice_matches = []
        choice_corrs = []
        gold_accs = []
        target_gold_accs = []
        for ex in batch["examples"]:
            if ex.choices:
                target_scores = target_model.score_choices_with_target(ex.text, ex.choices, max_prompt_len=max_prompt_len)
                pred_scores = target_model.score_choices_with_interpreter(interpreter, ex.text, ex.choices, max_prompt_len=max_prompt_len)
                gold_idx = ex.choices.index(ex.answer) if ex.answer in ex.choices else None
                cm = restricted_choice_metrics(target_scores, pred_scores, gold_idx)
                choice_matches.append(cm["choice_match_to_target"])
                choice_corrs.append(cm["choice_rank_corr"])
                if not np.isnan(cm["choice_acc_vs_gold"]):
                    gold_accs.append(cm["choice_acc_vs_gold"])
                if not np.isnan(cm["target_acc_vs_gold"]):
                    target_gold_accs.append(cm["target_acc_vs_gold"])

                example_artifacts.append({
                    "family": ex.family,
                    "source": ex.source,
                    "prompt": ex.text,
                    "choices": ex.choices,
                    "gold_answer": ex.answer,
                    "target_scores": target_scores,
                    "pred_scores": pred_scores,
                })

        metrics["choice_match_to_target"] = float(np.mean(choice_matches)) if choice_matches else float("nan")
        metrics["choice_rank_corr"] = float(np.mean(choice_corrs)) if choice_corrs else float("nan")
        metrics["choice_acc_vs_gold"] = float(np.mean(gold_accs)) if gold_accs else float("nan")
        metrics["target_choice_acc_vs_gold"] = float(np.mean(target_gold_accs)) if target_gold_accs else float("nan")
        rows.append(metrics)

    interpreter.train()
    return rows, example_artifacts
