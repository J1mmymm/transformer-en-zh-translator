---
language:
- en
- zh
license: apache-2.0
library_name: transformers
pipeline_tag: translation
tags:
- marian
- transformer
- translation
- en-zh
base_model: Helsinki-NLP/opus-mt-en-zh
metrics:
- bleu
---

# English-to-Chinese MarianMT fine-tune

This repository contains an English-to-Chinese MarianMT model fine-tuned from [`Helsinki-NLP/opus-mt-en-zh`](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh), plus the checkpoint of a smaller Transformer trained from scratch for comparison.

## Intended use

- Educational English-to-Chinese sentence translation
- Reproducing the accompanying Transformer/MarianMT comparison
- Local inference and further research fine-tuning

This is not a production translation service. It was trained on short conversational sentences and can mistranslate long, technical, safety-critical, or culturally sensitive text.

## Evaluation

The fixed course test split contained 2,991 sentence pairs. Scores were computed with SacreBLEU (`tokenize="zh"`), chrF++ and TER.

| Model | BLEU | chrF++ | TER | Exact match |
|---|---:|---:|---:|---:|
| Scratch Transformer | 20.5691 | 16.0240 | 66.1049 | 3.0090% |
| Base MarianMT (zero-shot) | 41.7833 | 28.7659 | 41.7261 | 2.1398% |
| Fine-tuned MarianMT | **52.4721** | **39.2654** | **30.8395** | **17.8536%** |

These scores describe this fixed local split only and should not be interpreted as general-domain translation quality.

## Training details

- Base model: `Helsinki-NLP/opus-mt-en-zh`
- Original course corpus: 26,918 training pairs, split into 25,573 train and 1,345 validation pairs
- Epochs: 4; selected checkpoint: epoch 3 by validation loss
- Optimizer: AdamW
- Learning rate: `2e-5` with cosine decay
- Micro-batch: 16; gradient accumulation: 2
- Beam size: 4
- Seed: 2026
- Training device: NVIDIA GeForce RTX 4060 Laptop GPU

The sentence content appears to derive from the Tatoeba/ManyThings English–Mandarin collection. The course files did not retain per-sentence attribution metadata, so the corpus itself is not redistributed here. See the project repository for the expected data format and reproducibility notes.

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "J1mmymm/transformer-en-zh-translator"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

inputs = tokenizer("Machine learning is useful.", return_tensors="pt")
outputs = model.generate(**inputs, num_beams=4, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Additional scratch checkpoint

`scratch/scratch_transformer.pt` contains the state dictionary, English and Chinese vocabularies, exact architecture/training configuration, best validation loss, and best epoch for the custom `nn.Transformer` model. Load it with `en_zh_translator.load_scratch_checkpoint` from the GitHub project.

## Limitations and bias

The model inherits limitations and biases from the base model and the short conversational fine-tuning corpus. References may have multiple valid translations, so exact-match is a particularly strict and incomplete metric. Human review is required before consequential use.

## Licensing and attribution

The base model is Apache-2.0. Project code and this derived model release are published under Apache-2.0. Dataset text is not included; users preparing their own Tatoeba/ManyThings split must follow its attribution requirements.
