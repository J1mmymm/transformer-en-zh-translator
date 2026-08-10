from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sacrebleu.metrics import BLEU, CHRF, TER
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATA_DIR = Path(os.environ.get("EN_ZH_DATA_DIR", ROOT / "data"))
MODEL_REL = os.environ.get("EN_ZH_BASE_MODEL", "Helsinki-NLP/opus-mt-en-zh")
OUTPUT_DIR = Path(os.environ.get("EN_ZH_OUTPUT_DIR", ROOT / "outputs"))
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CKPT_DIR = OUTPUT_DIR / "checkpoints"

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
SHOW_PROGRESS = os.environ.get("HW4_PROGRESS") == "1"


@dataclass
class RunConfig:
    seed: int = 2026
    val_ratio: float = 0.05
    scratch_src_vocab: int = 9000
    scratch_tgt_vocab: int = 5000
    scratch_max_src_len: int = 48
    scratch_max_tgt_len: int = 56
    scratch_batch_size: int = 128
    scratch_epochs: int = 24
    scratch_d_model: int = 256
    scratch_layers: int = 3
    scratch_heads: int = 4
    scratch_ffn: int = 1024
    scratch_dropout: float = 0.15
    scratch_lr: float = 3e-4
    scratch_patience: int = 6
    hf_batch_size: int = 32
    hf_train_batch_size: int = 16
    hf_grad_accum: int = 2
    hf_epochs: int = 4
    hf_lr: float = 2e-5
    hf_max_src_len: int = 64
    hf_max_tgt_len: int = 64
    hf_num_beams: int = 4
    hf_max_new_tokens: int = 64


class Vocab:
    def __init__(self, tokens: list[list[str]], max_size: int | None = None, min_freq: int = 1):
        counter: Counter[str] = Counter()
        for seq in tokens:
            counter.update(seq)
        words = [w for w, c in counter.most_common() if c >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIALS))]
        self.itos = SPECIALS + words
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(tok, self.unk_id) for tok in tokens[: max_len - 2])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars: list[str] = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            token = self.itos[idx] if 0 <= idx < len(self.itos) else UNK
            if token not in SPECIALS:
                chars.append(token)
        return "".join(chars)

    def to_json(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, data: dict) -> "Vocab":
        obj = cls.__new__(cls)
        obj.itos = list(data["itos"])
        obj.stoi = {token: idx for idx, token in enumerate(obj.itos)}
        obj.pad_id = obj.stoi[PAD]
        obj.bos_id = obj.stoi[BOS]
        obj.eos_id = obj.stoi[EOS]
        obj.unk_id = obj.stoi[UNK]
        return obj


EN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]")


def tokenize_en(text: str) -> list[str]:
    return EN_RE.findall(text.lower())


def tokenize_zh(text: str) -> list[str]:
    return [ch for ch in text.strip() if not ch.isspace()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, PRED_DIR, FIG_DIR, TABLE_DIR, CKPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src, tgt = line.split("\t", 1)
        pairs.append((src.strip(), tgt.strip()))
    return pairs


def train_val_split(pairs: list[tuple[str, str]], val_ratio: float, seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_idx = set(indices[:n_val])
    train = [pair for i, pair in enumerate(pairs) if i not in val_idx]
    val = [pair for i, pair in enumerate(pairs) if i in val_idx]
    return train, val


class ScratchDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], src_vocab: Vocab, tgt_vocab: Vocab, cfg: RunConfig):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        src, tgt = self.pairs[idx]
        src_ids = self.src_vocab.encode(tokenize_en(src), self.cfg.scratch_max_src_len)
        tgt_ids = self.tgt_vocab.encode(tokenize_zh(tgt), self.cfg.scratch_max_tgt_len)
        return src_ids, tgt_ids


def pad_batch(batch: list[tuple[list[int], list[int]]], src_pad: int, tgt_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    src_max = max(len(x[0]) for x in batch)
    tgt_max = max(len(x[1]) for x in batch)
    src = torch.full((len(batch), src_max), src_pad, dtype=torch.long)
    tgt = torch.full((len(batch), tgt_max), tgt_pad, dtype=torch.long)
    for i, (src_ids, tgt_ids) in enumerate(batch):
        src[i, : len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        tgt[i, : len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class ScratchTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_id: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_key_padding = src.eq(self.pad_id)
        tgt_key_padding = tgt_in.eq(self.pad_id)
        tgt_mask = torch.triu(
            torch.ones(
                (tgt_in.size(1), tgt_in.size(1)),
                dtype=torch.bool,
                device=tgt_in.device,
            ),
            diagonal=1,
        )
        src_emb = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
        )
        return self.generator(out)


def build_scratch_resources(train_pairs: list[tuple[str, str]], cfg: RunConfig) -> tuple[Vocab, Vocab]:
    src_tokens = [tokenize_en(src) for src, _ in train_pairs]
    tgt_tokens = [tokenize_zh(tgt) for _, tgt in train_pairs]
    src_vocab = Vocab(src_tokens, max_size=cfg.scratch_src_vocab, min_freq=1)
    tgt_vocab = Vocab(tgt_tokens, max_size=cfg.scratch_tgt_vocab, min_freq=1)
    return src_vocab, tgt_vocab


def train_scratch(train_pairs: list[tuple[str, str]], val_pairs: list[tuple[str, str]], cfg: RunConfig, device: torch.device, force: bool) -> tuple[ScratchTransformer, Vocab, Vocab, pd.DataFrame]:
    ckpt_path = CKPT_DIR / "scratch_transformer.pt"
    history_path = TABLE_DIR / "training_history_scratch.csv"
    if ckpt_path.exists() and history_path.exists() and not force:
        payload = torch.load(ckpt_path, map_location=device)
        src_vocab = Vocab.from_json(payload["src_vocab"])
        tgt_vocab = Vocab.from_json(payload["tgt_vocab"])
        model = ScratchTransformer(
            len(src_vocab),
            len(tgt_vocab),
            src_vocab.pad_id,
            cfg.scratch_d_model,
            cfg.scratch_heads,
            cfg.scratch_layers,
            cfg.scratch_ffn,
            cfg.scratch_dropout,
        ).to(device)
        model.load_state_dict(payload["model_state"])
        return model, src_vocab, tgt_vocab, pd.read_csv(history_path)

    src_vocab, tgt_vocab = build_scratch_resources(train_pairs, cfg)
    train_ds = ScratchDataset(train_pairs, src_vocab, tgt_vocab, cfg)
    val_ds = ScratchDataset(val_pairs, src_vocab, tgt_vocab, cfg)
    collate = lambda batch: pad_batch(batch, src_vocab.pad_id, tgt_vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.scratch_batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.scratch_batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = ScratchTransformer(
        len(src_vocab),
        len(tgt_vocab),
        src_vocab.pad_id,
        cfg.scratch_d_model,
        cfg.scratch_heads,
        cfg.scratch_layers,
        cfg.scratch_ffn,
        cfg.scratch_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.scratch_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scratch_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, cfg.scratch_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for src, tgt in tqdm(train_loader, desc=f"scratch epoch {epoch}/{cfg.scratch_epochs}", leave=False, disable=not SHOW_PROGRESS):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            non_pad = tgt_out.ne(tgt_vocab.pad_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
        scheduler.step()
        train_loss = total_loss / max(1, total_tokens)

        model.eval()
        val_total = 0.0
        val_tokens = 0
        with torch.inference_mode():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                non_pad = tgt_out.ne(tgt_vocab.pad_id).sum().item()
                val_total += loss.item() * non_pad
                val_tokens += non_pad
        val_loss = val_total / max(1, val_tokens)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "perplexity": math.exp(min(val_loss, 20.0)),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
        print(f"[scratch] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "src_vocab": src_vocab.to_json(),
                    "tgt_vocab": tgt_vocab.to_json(),
                    "config": asdict(cfg),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )
        elif epoch - best_epoch >= cfg.scratch_patience:
            print(f"[scratch] early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model, src_vocab, tgt_vocab, pd.DataFrame(history)


def greedy_decode_scratch(
    model: ScratchTransformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    texts: list[str],
    cfg: RunConfig,
    device: torch.device,
    batch_size: int = 128,
) -> list[str]:
    model.eval()
    preds: list[str] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="scratch decode", disable=not SHOW_PROGRESS):
            batch_texts = texts[start : start + batch_size]
            src_ids = [src_vocab.encode(tokenize_en(text), cfg.scratch_max_src_len) for text in batch_texts]
            src_max = max(len(x) for x in src_ids)
            src = torch.full((len(src_ids), src_max), src_vocab.pad_id, dtype=torch.long, device=device)
            for i, ids in enumerate(src_ids):
                src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            ys = torch.full((len(src_ids), 1), tgt_vocab.bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(len(src_ids), dtype=torch.bool, device=device)
            for _ in range(cfg.scratch_max_tgt_len):
                logits = model(src, ys)
                next_ids = logits[:, -1, :].argmax(dim=-1)
                next_ids = torch.where(finished, torch.full_like(next_ids, tgt_vocab.eos_id), next_ids)
                ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
                finished |= next_ids.eq(tgt_vocab.eos_id)
                if bool(finished.all()):
                    break
            for seq in ys[:, 1:].cpu().tolist():
                preds.append(tgt_vocab.decode(seq))
    return preds


def score_predictions(preds: list[str], refs: list[str]) -> dict[str, float]:
    refs_nested = [refs]
    bleu = BLEU(tokenize="zh").corpus_score(preds, refs_nested).score
    chrf = CHRF(word_order=2).corpus_score(preds, refs_nested).score
    ter = TER(normalized=True, asian_support=True).corpus_score(preds, refs_nested).score
    exact = sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs) * 100.0
    avg_len = sum(len(p) for p in preds) / len(preds)
    return {
        "BLEU": round(float(bleu), 4),
        "chrF++": round(float(chrf), 4),
        "TER": round(float(ter), 4),
        "ExactMatch(%)": round(float(exact), 4),
        "AvgPredChars": round(float(avg_len), 4),
    }


def write_prediction_csv(path: Path, english: list[str], refs: list[str], preds: list[str]) -> None:
    pd.DataFrame({"english": english, "reference": refs, "prediction": preds}).to_csv(path, index=False, encoding="utf-8-sig")


def generate_hf_predictions(
    model: AutoModelForSeq2SeqLM,
    tokenizer,
    src_texts: list[str],
    cfg: RunConfig,
    device: torch.device,
    desc: str,
) -> list[str]:
    model.eval()
    preds: list[str] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(src_texts), cfg.hf_batch_size), desc=desc, disable=not SHOW_PROGRESS):
            batch = src_texts[start : start + cfg.hf_batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=cfg.hf_max_src_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                num_beams=cfg.hf_num_beams,
                max_new_tokens=cfg.hf_max_new_tokens,
                early_stopping=True,
            )
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [p.strip() for p in preds]


def load_hf_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REL).to(device)
    return tokenizer, model


class HFDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def make_hf_collate(tokenizer, cfg: RunConfig):
    def collate(batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        src, tgt = zip(*batch)
        tokenized = tokenizer(
            list(src),
            text_target=list(tgt),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.hf_max_src_len,
            max_target_length=cfg.hf_max_tgt_len,
        )
        labels = tokenized["labels"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized["labels"] = labels
        return tokenized

    return collate


def train_hf_finetune(train_pairs: list[tuple[str, str]], val_pairs: list[tuple[str, str]], cfg: RunConfig, device: torch.device, force: bool):
    ckpt_path = CKPT_DIR / "marian_finetuned"
    ckpt_rel = "outputs/checkpoints/marian_finetuned"
    history_path = TABLE_DIR / "training_history_marian.csv"
    if ckpt_path.exists() and history_path.exists() and not force:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_rel)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_rel).to(device)
        return tokenizer, model, pd.read_csv(history_path)

    tokenizer, model = load_hf_model(device)
    model.config.use_cache = False
    train_loader = DataLoader(
        HFDataset(train_pairs),
        batch_size=cfg.hf_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_hf_collate(tokenizer, cfg),
    )
    val_loader = DataLoader(
        HFDataset(val_pairs),
        batch_size=cfg.hf_train_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_hf_collate(tokenizer, cfg),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.hf_lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(train_loader) / cfg.hf_grad_accum) * cfg.hf_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    global_step = 0
    for epoch in range(1, cfg.hf_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"marian finetune epoch {epoch}/{cfg.hf_epochs}", leave=False, disable=not SHOW_PROGRESS), start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = model(**batch).loss
                scaled_loss = loss / cfg.hf_grad_accum
            scaler.scale(scaled_loss).backward()
            if step % cfg.hf_grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            total_loss += loss.item() * batch["input_ids"].size(0)
            seen += batch["input_ids"].size(0)

        train_loss = total_loss / max(1, seen)
        model.eval()
        val_total = 0.0
        val_seen = 0
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="marian validation", leave=False, disable=not SHOW_PROGRESS):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                val_total += loss.item() * batch["input_ids"].size(0)
                val_seen += batch["input_ids"].size(0)
        val_loss = val_total / max(1, val_seen)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "perplexity": math.exp(min(val_loss, 20.0)),
            "global_step": global_step,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
        print(f"[marian] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_path), safe_serialization=True)
            tokenizer.save_pretrained(str(ckpt_path))
        elif epoch - best_epoch >= 2:
            print(f"[marian] early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    tokenizer = AutoTokenizer.from_pretrained(ckpt_rel)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_rel).to(device)
    return tokenizer, model, pd.DataFrame(history)


def make_plots(metrics_df: pd.DataFrame, scratch_history: pd.DataFrame, marian_history: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if not scratch_history.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(scratch_history["epoch"], scratch_history["train_loss"], marker="o", label="train")
        ax.plot(scratch_history["epoch"], scratch_history["val_loss"], marker="o", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Scratch Transformer Training Curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "scratch_training_curve.png", dpi=180)
        plt.close(fig)

    if not marian_history.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(marian_history["epoch"], marian_history["train_loss"], marker="o", label="train")
        ax.plot(marian_history["epoch"], marian_history["val_loss"], marker="o", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MarianMT Fine-tuning Curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "marian_finetune_curve.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = metrics_df.set_index("model")[["BLEU", "chrF++"]]
    plot_df.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Score")
    ax.set_title("Translation Quality on Test Set")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics_df["model"], metrics_df["TER"], color=["#7b8da6", "#4f9a8b", "#d27a46"])
    ax.set_ylabel("TER (lower is better)")
    ax.set_title("Translation Error Rate on Test Set")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ter_comparison.png", dpi=180)
    plt.close(fig)


def build_qualitative_examples(
    english: list[str],
    refs: list[str],
    scratch: list[str],
    zero: list[str],
    fine: list[str],
) -> pd.DataFrame:
    candidate_indices = [0, 1, 2, 9, 18, 30, 55, 100, 250, 500, 1000, 1500, 2200, 2800]
    rows = []
    for idx in candidate_indices:
        if idx < len(english):
            rows.append(
                {
                    "idx": idx,
                    "english": english[idx],
                    "reference": refs[idx],
                    "scratch": scratch[idx],
                    "zero_shot": zero[idx],
                    "finetuned": fine[idx],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    global DATA_DIR, MODEL_REL, OUTPUT_DIR, PRED_DIR, FIG_DIR, TABLE_DIR, CKPT_DIR

    os.chdir(ROOT)
    parser = argparse.ArgumentParser(
        description="Train and evaluate scratch Transformer and MarianMT English-to-Chinese models."
    )
    parser.add_argument("--force", action="store_true", help="rerun training and generation even if cached outputs exist")
    parser.add_argument("--quick", action="store_true", help="short smoke run for code validation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="directory containing UTF-8 train.txt and test.txt",
    )
    parser.add_argument(
        "--base-model",
        default=MODEL_REL,
        help="Hugging Face model ID or local MarianMT directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for checkpoints, predictions, tables and figures",
    )
    args = parser.parse_args()

    DATA_DIR = args.data_dir.expanduser().resolve()
    MODEL_REL = args.base_model
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    PRED_DIR = OUTPUT_DIR / "predictions"
    FIG_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR = OUTPUT_DIR / "tables"
    CKPT_DIR = OUTPUT_DIR / "checkpoints"

    for required in [DATA_DIR / "train.txt", DATA_DIR / "test.txt"]:
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing {required}. See data/README.md for the required format."
            )

    cfg = RunConfig()
    if args.quick:
        cfg.scratch_epochs = 1
        cfg.hf_epochs = 1
        cfg.scratch_batch_size = 64
        cfg.hf_batch_size = 8
        cfg.hf_train_batch_size = 4
        cfg.hf_grad_accum = 1
    ensure_dirs()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    train_pairs_all = read_pairs(DATA_DIR / "train.txt")
    test_pairs = read_pairs(DATA_DIR / "test.txt")
    if args.quick:
        train_pairs_all = train_pairs_all[:512]
        test_pairs = test_pairs[:64]
    train_pairs, val_pairs = train_val_split(train_pairs_all, cfg.val_ratio, cfg.seed)
    english = [src for src, _ in test_pairs]
    refs = [tgt for _, tgt in test_pairs]

    metadata = {
        "config": asdict(cfg),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    scratch_pred_path = PRED_DIR / "predictions_scratch.csv"
    if scratch_pred_path.exists() and not args.force:
        scratch_df = pd.read_csv(scratch_pred_path)
        scratch_preds = scratch_df["prediction"].fillna("").astype(str).tolist()
        scratch_history = pd.read_csv(TABLE_DIR / "training_history_scratch.csv")
    else:
        scratch_model, src_vocab, tgt_vocab, scratch_history = train_scratch(train_pairs, val_pairs, cfg, device, force=args.force)
        scratch_preds = greedy_decode_scratch(scratch_model, src_vocab, tgt_vocab, english, cfg, device)
        write_prediction_csv(scratch_pred_path, english, refs, scratch_preds)

    zero_pred_path = PRED_DIR / "predictions_zeroshot.csv"
    if zero_pred_path.exists() and not args.force:
        zero_preds = pd.read_csv(zero_pred_path)["prediction"].fillna("").astype(str).tolist()
    else:
        tokenizer, zero_model = load_hf_model(device)
        zero_preds = generate_hf_predictions(zero_model, tokenizer, english, cfg, device, "marian zero-shot decode")
        write_prediction_csv(zero_pred_path, english, refs, zero_preds)
        del zero_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fine_pred_path = PRED_DIR / "predictions_finetuned.csv"
    if fine_pred_path.exists() and (TABLE_DIR / "training_history_marian.csv").exists() and not args.force:
        fine_preds = pd.read_csv(fine_pred_path)["prediction"].fillna("").astype(str).tolist()
        marian_history = pd.read_csv(TABLE_DIR / "training_history_marian.csv")
    else:
        ft_tokenizer, ft_model, marian_history = train_hf_finetune(train_pairs, val_pairs, cfg, device, force=args.force)
        fine_preds = generate_hf_predictions(ft_model, ft_tokenizer, english, cfg, device, "marian finetuned decode")
        write_prediction_csv(fine_pred_path, english, refs, fine_preds)

    metrics_rows = []
    for model_name, preds in [
        ("Scratch Transformer", scratch_preds),
        ("MarianMT zero-shot", zero_preds),
        ("MarianMT fine-tuned", fine_preds),
    ]:
        row = {"model": model_name}
        row.update(score_predictions(preds, refs))
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(TABLE_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "metrics_summary.json").write_text(metrics_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    examples_df = build_qualitative_examples(english, refs, scratch_preds, zero_preds, fine_preds)
    examples_df.to_csv(TABLE_DIR / "qualitative_examples.csv", index=False, encoding="utf-8-sig")
    make_plots(metrics_df, scratch_history, marian_history)
    print(metrics_df.to_string(index=False))
    print(f"[done] outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
