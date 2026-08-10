from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
EN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]")


def tokenize_en(text: str) -> list[str]:
    return EN_RE.findall(text.lower())


def tokenize_zh(text: str) -> list[str]:
    return [char for char in text.strip() if not char.isspace()]


class Vocab:
    def __init__(
        self,
        tokens: list[list[str]],
        max_size: int | None = None,
        min_freq: int = 1,
    ) -> None:
        counter: Counter[str] = Counter()
        for sequence in tokens:
            counter.update(sequence)
        words = [word for word, count in counter.most_common() if count >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIALS))]
        self.itos = SPECIALS + words
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(token, self.unk_id) for token in tokens[: max_len - 2])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        characters: list[str] = []
        for index in ids:
            if index == self.eos_id:
                break
            if index in (self.pad_id, self.bos_id):
                continue
            token = self.itos[index] if 0 <= index < len(self.itos) else UNK
            if token not in SPECIALS:
                characters.append(token)
        return "".join(characters)

    def to_json(self) -> dict[str, list[str]]:
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Vocab":
        vocab = cls.__new__(cls)
        vocab.itos = list(data["itos"])
        vocab._rebuild_index()
        return vocab


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(inputs + self.pe[:, : inputs.size(1)])


class ScratchTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.15,
    ) -> None:
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
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
        )
        return self.generator(output)


def load_scratch_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ScratchTransformer, Vocab, Vocab, dict[str, Any]]:
    """Load the released scratch checkpoint and rebuild its exact architecture."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    src_vocab = Vocab.from_json(payload["src_vocab"])
    tgt_vocab = Vocab.from_json(payload["tgt_vocab"])
    config = dict(payload.get("config", {}))
    model = ScratchTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        pad_id=src_vocab.pad_id,
        d_model=int(config.get("scratch_d_model", 256)),
        nhead=int(config.get("scratch_heads", 4)),
        num_layers=int(config.get("scratch_layers", 3)),
        dim_feedforward=int(config.get("scratch_ffn", 1024)),
        dropout=float(config.get("scratch_dropout", 0.15)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab, payload


@torch.inference_mode()
def translate_scratch(
    model: ScratchTransformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    text: str,
    device: str | torch.device = "cpu",
    max_src_len: int = 48,
    max_tgt_len: int = 56,
) -> str:
    src_ids = src_vocab.encode(tokenize_en(text), max_src_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    generated = torch.full(
        (1, 1), tgt_vocab.bos_id, dtype=torch.long, device=device
    )
    finished = False
    for _ in range(max_tgt_len):
        logits = model(src, generated)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], device=device)], dim=1
        )
        if next_id == tgt_vocab.eos_id:
            finished = True
            break
    token_ids = generated[0, 1:].detach().cpu().tolist()
    if not finished:
        token_ids = token_ids[:max_tgt_len]
    return tgt_vocab.decode(token_ids)
