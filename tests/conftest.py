"""Shared fixtures.

`base_model` is a tiny randomly initialised BERT built on the fly, so tests
that use it need no network access. Its embeddings carry no meaning; tests
that need meaningful geometry should supply their own vectors.
"""

import pytest
import torch

WORDS = ("great terrible okay product service love hate awful fine good bad "
         "slow fast cheap broke works the a is it very").split()


@pytest.fixture(scope="session")
def base_model(tmp_path_factory):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    path = tmp_path_factory.mktemp("tiny_bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    torch.manual_seed(0)
    BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64,
    )).save_pretrained(path)
    BertTokenizerFast(str(path / "vocab.txt"), do_lower_case=True).save_pretrained(path)
    return str(path)


@pytest.fixture
def new_classifier(base_model):
    """Factory for an untrained classifier on the tiny offline model."""
    from adaptive_classifier import AdaptiveClassifier

    def make(**config):
        return AdaptiveClassifier(base_model, config=config, use_onnx=False, device="cpu")
    return make
