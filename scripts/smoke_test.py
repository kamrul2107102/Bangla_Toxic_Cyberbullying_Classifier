import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.models.tfidf_lr import TfidfLogRegModel
from src.models.naive_bayes import MultiLabelMultinomialNB
from src.models.word2vec_lr import Word2VecLogRegModel
from src.metrics import multilabel_metrics
from src.labels import to_arrays

rows = [
    {"text":"তুই খুব বাজে মানুষ", "source":"sample", "hate_speech":0,"sexist":0,"threat":0,"bullying":1,"toxic":1,"hate_speech_mask":1,"sexist_mask":1,"threat_mask":1,"bullying_mask":1,"toxic_mask":1,"context_text":""},
    {"text":"তোমাকে মেরে ফেলব", "source":"sample", "hate_speech":0,"sexist":0,"threat":1,"bullying":0,"toxic":1,"hate_speech_mask":1,"sexist_mask":1,"threat_mask":1,"bullying_mask":1,"toxic_mask":1,"context_text":""},
    {"text":"তুমি আজ খুব ভালো করেছো", "source":"sample", "hate_speech":0,"sexist":0,"threat":0,"bullying":0,"toxic":0,"hate_speech_mask":1,"sexist_mask":1,"threat_mask":1,"bullying_mask":1,"toxic_mask":1,"context_text":""},
    {"text":"ওই মেয়েটা কিছুই পারে না", "source":"sample", "hate_speech":0,"sexist":1,"threat":0,"bullying":1,"toxic":1,"hate_speech_mask":1,"sexist_mask":1,"threat_mask":1,"bullying_mask":1,"toxic_mask":1,"context_text":""},
]
df=pd.DataFrame(rows)
y,mask=to_arrays(df.to_dict("records"))
texts=df.text.tolist()
for model in [TfidfLogRegModel(max_features=500,min_df=1,epochs=3), MultiLabelMultinomialNB(), Word2VecLogRegModel(dim=16,max_vocab=100,epochs_w2v=1,epochs_lr=3,min_count=1)]:
    print("Testing", model.__class__.__name__)
    model.fit(texts,y,mask)
    p=model.predict_proba([texts[0],"তুমি ভালো"])
    print(p)
    assert p.shape==(2,5)
print("Smoke test passed")
