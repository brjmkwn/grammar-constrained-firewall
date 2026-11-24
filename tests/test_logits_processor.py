import torch

from app.core.compiler import compiler
from app.core.masker import FSMLogitsProcessor, TokenLogitMasker, default_vocab


def test_pytorch_masks_against_each_generated_prefix():
    grammar = compiler.compile_schema_to_fsm({"type": "string", "enum": ["a", "b"]})
    masker = TokenLogitMasker(fsm=grammar, vocab=default_vocab)
    processor = FSMLogitsProcessor(masker)
    open_quote = default_vocab.encode('"')[0]
    size = default_vocab.vocab_size
    prompt = torch.tensor([[open_quote, default_vocab.encode("prompt")[0]]])
    scores = torch.ones((1, size))

    initial = processor(prompt, scores.clone())
    assert torch.isneginf(initial[0, 0])
    assert initial[0, open_quote] == 1

    prefix = torch.cat((prompt, torch.tensor([[open_quote]])), dim=1)
    next_scores = processor(prefix, scores.clone())
    assert torch.isneginf(next_scores[0, open_quote])
    assert torch.isfinite(next_scores).any()
