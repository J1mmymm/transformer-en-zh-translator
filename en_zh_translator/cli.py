from __future__ import annotations

import argparse

import torch

DEFAULT_MODEL = "J1mmymm/transformer-en-zh-translator"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate English text into Chinese with the released MarianMT model."
    )
    parser.add_argument("text", nargs="*", help="English text. If omitted, read one line from stdin.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = " ".join(args.text).strip()
    if not text:
        text = input("English: ").strip()
    if not text:
        raise SystemExit("No input text provided.")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            early_stopping=True,
        )
    print(tokenizer.decode(generated[0], skip_special_tokens=True).strip())

