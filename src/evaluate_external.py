from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from .config import ARTIFACT_DIR, LABELS
from .predict import load_model


def binary_prf(y, p):
    tp=int(((y==1)&(p==1)).sum()); fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default='data/processed/external_banglatoco.csv')
    ap.add_argument('--threshold', type=float, default=0.5)
    args=ap.parse_args()
    df=pd.read_csv(args.data)
    model=load_model(args.model)
    probs=model.predict_proba(df['text'].fillna('').astype(str).tolist())
    j=LABELS.index('toxic')
    y=df['toxic'].astype(int).to_numpy()
    pred=(probs[:,j]>=args.threshold).astype(int)
    p,r,f=binary_prf(y,pred)
    out={'model':args.model,'dataset':args.data,'support':int(len(df)),'toxic_precision':p,'toxic_recall':r,'toxic_f1':f,'threshold':args.threshold}
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
