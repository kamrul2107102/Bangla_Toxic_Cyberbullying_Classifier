from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

from .config import LABELS, RAW_DIR, PROCESSED_DIR, SEED
from .preprocessing import normalize_text
from .labels import blank_labels, clean_binary


def pick_col(df: pd.DataFrame, candidates: list[str], required=True):
    normalized = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in normalized:
            return normalized[key]
    # Fuzzy fallback: candidate substring in normalized column name or vice-versa.
    for key, original in normalized.items():
        if not key:
            continue
        for cand in candidates:
            cand_key = re.sub(r"[^a-z0-9]", "", cand.lower())
            if cand_key and (cand_key in key or key in cand_key):
                return original
    if required:
        raise ValueError(f"Could not find a column among {candidates}. Available: {list(df.columns)}")
    return None


def source_name(path: Path) -> str:
    n = path.name.lower().replace("_", "-")
    if "multi" in n and "toxic" in n:
        return "multi_toxic"
    if "bidwesh" in n:
        return "bidwesh"
    if "toco" in n or "to-co" in n or "context" in n or "aware" in n:
        return "banglatoco"
    if "cyber" in n or "bully" in n:
        return "cyberbullying"
    return "generic"


def base_row(text: str, source: str, context_text: str = ""):
    row = {"text": text, "source": source, "context_text": context_text}
    row.update(blank_labels())
    return row


def ingest_multi_toxic(df: pd.DataFrame):
    text_col = pick_col(df, ["text", "comment", "sentence"])
    rows = []
    source = "multi_toxic"
    label_cols = {
        "vulgar": pick_col(df, ["vulgar"], False),
        "hate": pick_col(df, ["hate"], False),
        "religious": pick_col(df, ["religious"], False),
        "threat": pick_col(df, ["threat"], False),
        "troll": pick_col(df, ["troll"], False),
        "insult": pick_col(df, ["insult"], False),
    }
    for _, r in df.iterrows():
        text = str(r[text_col])
        out = base_row(text, source)
        vals = {k: clean_binary(r[c]) if c else None for k, c in label_cols.items()}
        toxic_known = any(v is not None for v in vals.values())
        if toxic_known:
            toxic = int(any(v == 1 for v in vals.values()))
            out["toxic"], out["toxic_mask"] = toxic, 1
            out["hate_speech"] = int((vals.get("hate") == 1) or (vals.get("religious") == 1))
            out["hate_speech_mask"] = int(vals.get("hate") is not None or vals.get("religious") is not None)
            out["threat"] = int(vals.get("threat") == 1)
            out["threat_mask"] = int(vals.get("threat") is not None)
            out["bullying"] = int((vals.get("troll") == 1) or (vals.get("insult") == 1))
            out["bullying_mask"] = int(vals.get("troll") is not None or vals.get("insult") is not None)
            # No explicit sexist annotation in this source: keep masked as unknown.
        rows.append(out)
    return rows


def ingest_hate(df: pd.DataFrame, source="hate"):
    text_col = pick_col(df, ["sentence", "text", "comment"])
    hate_col = pick_col(df, ["hate", "label", "hs"])
    rows = []
    for _, r in df.iterrows():
        hv = clean_binary(r[hate_col])
        if hv is None:
            continue
        out = base_row(str(r[text_col]), source)
        out["hate_speech"], out["hate_speech_mask"] = hv, 1
        rows.append(out)
    return rows


def ingest_cyberbullying(df: pd.DataFrame):
    text_col = pick_col(df, ["text", "comment", "sentence", "content", "description"])
    label_col = pick_col(df, ["label", "category", "class", "type"])
    rows = []
    for _, r in df.iterrows():
        raw = str(r[label_col]).lower().strip()
        out = base_row(str(r[text_col]), "cyberbullying")
        # Source categories: political, sexual, troll, threat, neutral.
        if raw:
            out["toxic"], out["toxic_mask"] = int(raw != "neutral"), 1
            out["threat"], out["threat_mask"] = int("threat" in raw), 1
            out["bullying"], out["bullying_mask"] = int("troll" in raw or "sexual" in raw), 1
            out["sexist"], out["sexist_mask"] = int("sexual" in raw), 1
        rows.append(out)
    return rows


def ingest_bidwesh(df: pd.DataFrame):
    text_col = pick_col(df, ["sentence", "text", "comment", "target comment", "description"], required=True)
    hate_col = pick_col(df, ["hate", "hate speech", "hs", "label"], required=False)
    type_col = pick_col(df, ["type", "hate type", "hate_type", "category"], required=False)
    rows = []
    for _, r in df.iterrows():
        out = base_row(str(r[text_col]), "bidwesh")
        if hate_col:
            hv = clean_binary(r[hate_col])
            if hv is not None:
                out["hate_speech"], out["hate_speech_mask"] = hv, 1
                out["toxic"], out["toxic_mask"] = hv, 1
        if type_col and str(r[type_col]).strip():
            typ = str(r[type_col]).lower()
            out["sexist"], out["sexist_mask"] = int("gender" in typ or "sex" in typ), 1
            out["threat"], out["threat_mask"] = int("violence" in typ or "threat" in typ or "callto" in typ), 1
        rows.append(out)
    return rows


def ingest_toco(df: pd.DataFrame):
    text_col = pick_col(df, ["target comment", "target_comment", "comment", "text", "description"])
    label_col = pick_col(df, ["label", "class", "toxicity"])
    context_cols = [
        pick_col(df, ["news title", "news_title", "title"], False),
        pick_col(df, ["metadata", "meta"], False),
        pick_col(df, ["predecessor comment", "predecessor_comment", "predecessor", "previous comment"], False),
        pick_col(df, ["successor comment", "successor_comment", "successor", "next comment"], False),
    ]
    rows = []
    for _, r in df.iterrows():
        label = str(r[label_col]).strip().lower()
        out = base_row(str(r[text_col]), "banglatoco")
        out["toxic"], out["toxic_mask"] = int(label.startswith("toxic") or label in {"1", "true"}), 1
        ctx = [str(r[c]) for c in context_cols if c and pd.notna(r[c]) and str(r[c]).strip() and str(r[c]).strip() != "N/A"]
        out["context_text"] = " ".join(ctx)
        rows.append(out)
    return rows


def ingest_csv(path: Path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    src = source_name(path)
    if src == "multi_toxic":
        return ingest_multi_toxic(df)
    if src == "cyberbullying":
        return ingest_cyberbullying(df)
    if src == "bidwesh":
        return ingest_bidwesh(df)
    if src == "banglatoco":
        return ingest_toco(df)
    # Generic fallback: hate-like or binary label column.
    try:
        return ingest_hate(df, source="generic")
    except Exception as exc:
        raise ValueError(f"Could not auto-ingest {path.name}. Please add a source-specific adapter. {exc}")


def prepare(input_dir=RAW_DIR, output_path=PROCESSED_DIR / "merged_dataset.csv"):
    input_dir = Path(input_dir)
    files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xls")))
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {input_dir}")
    rows = []
    for path in files:
        try:
            part = ingest_csv(path)
            print(f"[OK] {path.name}: {len(part)} rows")
            rows.extend(part)
        except Exception as exc:
            print(f"[SKIP] {path.name}: {exc}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No dataset rows were ingested.")
    df["norm_text"] = df["text"].map(normalize_text)
    df = df[df["norm_text"].str.len() > 0].copy()
    # Global deduplication prevents exact-text leakage across source-derived datasets.
    df = df.drop_duplicates(subset=["norm_text"], keep="first").reset_index(drop=True)
    # Drop helper; keep raw text and context.
    df = df.drop(columns=["norm_text"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Bangla-ToCo is kept as a clean external/context-aware test set rather than silently
    # mixing its binary annotation with all multi-label training targets.
    external_toco = df[df["source"] == "banglatoco"].copy()
    train_df = df[df["source"] != "banglatoco"].copy()
    df.to_csv(output_path.parent / "all_sources_dataset.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    if not external_toco.empty:
        external_toco.to_csv(output_path.parent / "external_banglatoco.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(train_df)} training-corpus rows to {output_path}")
    print(f"Saved {len(df)} all-source rows to {output_path.parent / 'all_sources_dataset.csv'}")
    if not external_toco.empty:
        print(f"Saved {len(external_toco)} Bangla-ToCo external rows to {output_path.parent / 'external_banglatoco.csv'}")
    print("Training source counts:\n", train_df["source"].value_counts())
    return train_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(RAW_DIR))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "merged_dataset.csv"))
    args = parser.parse_args()
    prepare(args.input_dir, args.output)
