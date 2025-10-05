#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework Kotara (beta)
==========================

Avaliador **generalista** de Perplexity (PPL) com **fusão adaptativa** entre dois modelos *causal LM* da Hugging Face(ou modelos alternativos).

✔ Suporta **qualquer arquitetura** compatível com `AutoModelForCausalLM` (inclui modelos custom com `trust_remote_code=True`).
✔ Estratégias de fusão disponível nessa versao: `average`, `poe`, `entropy`, `gap`.
✔ Recomendado usar **um *tokenizer compartilhado*** (mesmo vocabulário) para fusão correta.
✔ GOKU + VEGETA + POTARA = VEGETTO/ LLM1 + LLM2 + KOTARA = FUSED LLM

Autor: Gabriel (MadrasLe) — 2025  |  Licença: Apache 2.0 | GTLM Research
"""
import argparse
import json
import math
import os
from typing import Dict, Tuple, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================
# Utilidades numéricas gerais
# =============================

def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Entropia de Shannon por posição (dim=-1)."""
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def top1_gap_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Diferença p(top1) - p(top2) por posição (maior → mais confiante)."""
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    return sorted_probs[..., 0] - sorted_probs[..., 1]


def log_softmax_(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def softmax_(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


# =============================
# Fusão de logits / log-probs
# =============================

def fuse_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    strategy: str = "entropy",
    temp_a: float = 1.0,
    temp_b: float = 1.0,
    poe_beta_a: float = 0.5,
    poe_beta_b: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combina previsões de dois modelos conforme a estratégia.

    Retorna:
        fused (torch.Tensor):
            • Para `poe`: **log-probabilities** (saída já em log-softmax).
            • Para demais estratégias: **logits**.
        stats (dict): métricas auxiliares (pesos médios, etc.).
    """
    stats: Dict[str, float] = {}

    logits_a = logits_a / max(temp_a, 1e-9)
    logits_b = logits_b / max(temp_b, 1e-9)

    if strategy == "average":
        fused = (logits_a + logits_b) / 2.0
        stats.update({"w_a_mean": 0.5, "w_b_mean": 0.5})
        return fused, stats

    if strategy == "poe":
        la = log_softmax_(logits_a)
        lb = log_softmax_(logits_b)
        fused_log_probs = poe_beta_a * la + poe_beta_b * lb
        stats.update({"beta_a": float(poe_beta_a), "beta_b": float(poe_beta_b)})
        return fused_log_probs, stats

    # Estratégias baseadas em confiança (pesos adaptativos por posição)
    pa = softmax_(logits_a)
    pb = softmax_(logits_b)

    if strategy == "entropy":
        ua = entropy_from_probs(pa)  # menor = mais confiante
        ub = entropy_from_probs(pb)
        wa = 1.0 / (ua + 1e-9)
        wb = 1.0 / (ub + 1e-9)
    elif strategy == "gap":
        wa = top1_gap_from_probs(pa)
        wb = top1_gap_from_probs(pb)
    else:
        raise ValueError(f"Estratégia de fusão desconhecida: {strategy}")

    total = wa + wb + 1e-9
    alpha_a = wa / total
    alpha_b = wb / total

    stats.update({
        "w_a_mean": float(alpha_a.mean().item()),
        "w_b_mean": float(alpha_b.mean().item()),
    })

    fused = alpha_a.unsqueeze(-1) * logits_a + alpha_b.unsqueeze(-1) * logits_b
    return fused, stats


# =============================
# Loss / PPL helpers
# =============================

def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Negative Log-Likelihood a partir de logits (CrossEntropy)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return float(loss.item()) * shift_labels.numel()


def nll_from_log_probs(log_probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Negative Log-Likelihood diretamente de log-probabilities (já normalizadas)."""
    shift_log_probs = log_probs[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    flat_log_probs = shift_log_probs.view(-1, shift_log_probs.size(-1))
    flat_labels = shift_labels.view(-1)
    picked = flat_log_probs[torch.arange(flat_labels.numel(), device=flat_labels.device), flat_labels]
    nll_sum = -picked.mean().item() * flat_labels.numel()
    return float(nll_sum)


# =============================
# Leitura em sliding window e avaliação
# =============================

@torch.no_grad()
def compute_ppl_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    device: str = "cuda",
    max_length: int = 1024,
    stride: int = 1024,
    chunk_bytes: int = 1024 * 1024,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    """PPL de um único modelo (baseline)."""
    model.eval()
    total_tokens = 0
    total_nll = 0.0

    with open(dataset_path, "r", encoding="utf-8") as f:
        pbar = tqdm(desc=f"Kotara Baseline: {os.path.basename(dataset_path)}", unit="MB")
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            pbar.update(len(chunk.encode("utf-8")) / (1024 * 1024))

            enc = tokenizer(chunk, return_tensors="pt")
            input_ids = enc.input_ids.to(device)

            T = input_ids.size(1)
            for start in range(0, T, stride):
                end = min(start + max_length, T)
                if end - start < 2:
                    continue
                ids = input_ids[:, start:end]
                targets = ids.clone()

                with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=autocast_dtype) if autocast_dtype else torch.no_grad():
                    logits = model(input_ids=ids).logits
                nll = nll_from_logits(logits, targets)
                tokens = (end - start - 1)
                total_nll += nll
                total_tokens += tokens
        pbar.close()

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return ppl


@torch.no_grad()
def compute_ppl_fused(
    model_a: AutoModelForCausalLM,
    model_b: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    device: str = "cuda",
    max_length: int = 1024,
    stride: int = 1024,
    chunk_bytes: int = 1024 * 1024,
    strategy: str = "entropy",
    temp_a: float = 1.0,
    temp_b: float = 1.0,
    poe_beta_a: float = 0.5,
    poe_beta_b: float = 0.5,
    autocast_dtype: torch.dtype | None = None,
) -> Tuple[float, Dict[str, Any]]:
    """Calcula PPL da fusão adaptativa em um arquivo de texto.

    Observação: assume **tokenizer/vocabulário compartilhado** entre A e B.
    """
    model_a.eval(); model_b.eval()

    total_tokens = 0
    total_nll = 0.0
    stats_accum = {"w_a_mean_sum": 0.0, "w_b_mean_sum": 0.0, "steps": 0}

    with open(dataset_path, "r", encoding="utf-8") as f:
        pbar = tqdm(desc=f"Kotara Fused: {os.path.basename(dataset_path)} [{strategy}]", unit="MB")
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            pbar.update(len(chunk.encode("utf-8")) / (1024 * 1024))

            enc = tokenizer(chunk, return_tensors="pt")
            input_ids = enc.input_ids.to(device)

            T = input_ids.size(1)
            for start in range(0, T, stride):
                end = min(start + max_length, T)
                if end - start < 2:
                    continue
                ids = input_ids[:, start:end]
                targets = ids.clone()

                with torch.autocast(device_type=device if device != "cpu" else "cpu", dtype=autocast_dtype) if autocast_dtype else torch.no_grad():
                    logits_a = model_a(input_ids=ids).logits
                    logits_b = model_b(input_ids=ids).logits

                fused, s = fuse_logits(
                    logits_a, logits_b,
                    strategy=strategy,
                    temp_a=temp_a, temp_b=temp_b,
                    poe_beta_a=poe_beta_a, poe_beta_b=poe_beta_b,
                )

                if strategy == "poe":
                    nll = nll_from_log_probs(fused, targets)
                else:
                    nll = nll_from_logits(fused, targets)

                tokens = (end - start - 1)
                total_nll += nll
                total_tokens += tokens

                if "w_a_mean" in s:
                    stats_accum["w_a_mean_sum"] += s["w_a_mean"]
                    stats_accum["w_b_mean_sum"] += s["w_b_mean"]
                    stats_accum["steps"] += 1
        pbar.close()

    ppl = math.exp(total_nll / max(total_tokens, 1))
    final_stats = {
        "tokens": int(total_tokens),
        "w_a_mean": (stats_accum["w_a_mean_sum"] / stats_accum["steps"]) if stats_accum["steps"] else None,
        "w_b_mean": (stats_accum["w_b_mean_sum"] / stats_accum["steps"]) if stats_accum["steps"] else None,
    }
    return ppl, final_stats


# =============================
# CLI / Main
# =============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kotara (beta): Avaliador de PPL com fusão adaptativa entre dois LLMs (HF)")

    # Modelos (HF names ou caminhos locais)
    p.add_argument("--model_a", type=str, required=True, help="HuggingFace id ou caminho local do Modelo A")
    p.add_argument("--model_b", type=str, required=True, help="HuggingFace id ou caminho local do Modelo B")

    # Tokenizer compartilhado (recomendado). Se ausente, usa tokenizer do modelo A
    p.add_argument("--tokenizer", type=str, default=None, help="HF id/caminho do tokenizer compartilhado")

    # Arquivo de avaliação
    p.add_argument("--dataset_path", type=str, required=True, help="Arquivo de texto para avaliação de PPL")

    # Janelamento / leitura
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--chunk_bytes", type=int, default=1024 * 1024)

    # Estratégia de fusão
    p.add_argument("--strategy", type=str, default="entropy", choices=["average", "poe", "entropy", "gap"])
    p.add_argument("--temp_a", type=float, default=1.0)
    p.add_argument("--temp_b", type=float, default=1.0)
    p.add_argument("--poe_beta_a", type=float, default=0.5)
    p.add_argument("--poe_beta_b", type=float, default=0.5)

    # Dispositivo / dtype
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Auto usa bf16>fp16>fp32 conforme suporte")

    # trust_remote_code para modelos custom
    p.add_argument("--trust_remote_code", action="store_true")

    # Saída
    p.add_argument("--save_metrics", type=str, default=None, help="Caminho .json para salvar métricas")

    return p.parse_args()


def pick_dtype(device: str, dtype_flag: str) -> torch.dtype | None:
    if dtype_flag == "auto":
        if device != "cpu" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return None
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_flag]


def load_model(model_ref: str, device: str, trust_remote_code: bool) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype="auto",
        device_map=None,  # manual
        trust_remote_code=trust_remote_code,
    )
    return model.to(device)


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)

    # Tokenizer compartilhado
    tok_ref = args.tokenizer if args.tokenizer is not None else args.model_a
    tokenizer = AutoTokenizer.from_pretrained(tok_ref, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Carrega modelos (qualquer arquitetura HF causal LM)
    print("Carregando modelos…")
    model_a = load_model(args.model_a, args.device, args.trust_remote_code)
    model_b = load_model(args.model_b, args.device, args.trust_remote_code)

    # Verificação leve de compatibilidade de vocabulário
    if model_a.get_input_embeddings().weight.size(0) != model_b.get_input_embeddings().weight.size(0):
        print("[AVISO] Vocabulários de A e B parecem ter tamanhos diferentes. Recomenda-se tokenizer compartilhado.")

    # Dtype para autocast (não força conversão de pesos; só o compute)
    autocast_dtype = pick_dtype(args.device, args.dtype)

    # Baselines
    print("
Calculando baseline (Modelo A)…")
    ppl_a = compute_ppl_model(
        model_a, tokenizer, args.dataset_path,
        device=args.device, max_length=args.max_length, stride=args.stride, chunk_bytes=args.chunk_bytes,
        autocast_dtype=autocast_dtype,
    )

    print("Calculando baseline (Modelo B)…")
    ppl_b = compute_ppl_model(
        model_b, tokenizer, args.dataset_path,
        device=args.device, max_length=args.max_length, stride=args.stride, chunk_bytes=args.chunk_bytes,
        autocast_dtype=autocast_dtype,
    )

    # Fusão
    print("
Calculando PPL fundida (Kotara)…")
    ppl_fused, stats = compute_ppl_fused(
        model_a, model_b, tokenizer, args.dataset_path,
        device=args.device, max_length=args.max_length, stride=args.stride, chunk_bytes=args.chunk_bytes,
        strategy=args.strategy, temp_a=args.temp_a, temp_b=args.temp_b,
        poe_beta_a=args.poe_beta_a, poe_beta_b=args.poe_beta_b,
        autocast_dtype=autocast_dtype,
    )

    print("
" + "=" * 68)
    print("🏆  RESULTADOS — Framework do Kotara (beta)")
    print("=" * 68)
    print(f"Baseline A  → PPL: {ppl_a:.4f}")
    print(f"Baseline B  → PPL: {ppl_b:.4f}")
    print(f"Fusão [{args.strategy}] → PPL: {ppl_fused:.4f}")
    if stats["w_a_mean"] is not None:
        print(f"Pesos médios: w_A={stats['w_a_mean']:.3f} | w_B={stats['w_b_mean']:.3f}")
    print("=" * 68)

    if args.save_metrics:
        out = {
            "baseline_ppl_a": ppl_a,
            "baseline_ppl_b": ppl_b,
            "fused_ppl": ppl_fused,
            "strategy": args.strategy,
            "temps": {"a": args.temp_a, "b": args.temp_b},
            "poe_betas": {"a": args.poe_beta_a, "b": args.poe_beta_b},
            "weights": {"w_a_mean": stats["w_a_mean"], "w_b_mean": stats["w_b_mean"]},
            "config": {
                "tokenizer": tok_ref,
                "max_length": args.max_length,
                "stride": args.stride,
                "chunk_bytes": args.chunk_bytes,
                "dtype": args.dtype,
                "device": args.device,
                "model_a": args.model_a,
                "model_b": args.model_b,
                "trust_remote_code": args.trust_remote_code,
            },
        }
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"
✅ Métricas salvas em: {args.save_metrics}")


if __name__ == "__main__":
    main()
