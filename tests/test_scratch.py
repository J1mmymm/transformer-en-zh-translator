import torch

from en_zh_translator.scratch import ScratchTransformer, Vocab, tokenize_en, tokenize_zh


def test_tokenizers_and_vocab_round_trip() -> None:
    assert tokenize_en("I'm OK.") == ["i'm", "ok", "."]
    assert tokenize_zh("你 好。") == ["你", "好", "。"]
    vocab = Vocab([["你", "好"], ["好", "。"]])
    assert vocab.decode(vocab.encode(["你", "好"], max_len=8)) == "你好"


def test_scratch_transformer_forward_shape() -> None:
    model = ScratchTransformer(
        src_vocab_size=12,
        tgt_vocab_size=14,
        pad_id=0,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    src = torch.tensor([[1, 4, 2, 0]])
    tgt = torch.tensor([[1, 5, 6]])
    logits = model(src, tgt)
    assert logits.shape == (1, 3, 14)

