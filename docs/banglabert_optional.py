"""Optional benchmark only. This is intentionally separate from the proposal-compliant core."""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "csebuetnlp/banglabert"


def build_model(num_labels=5):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


if __name__ == "__main__":
    print("Optional benchmark: BanglaBERT")
    print("This uses a pretrained Bengali ELECTRA checkpoint and therefore is NOT proposal-compliant when the no-pretrained-model rule is enforced.")
    tokenizer, model = build_model()
    print(type(model).__name__, "loaded")
