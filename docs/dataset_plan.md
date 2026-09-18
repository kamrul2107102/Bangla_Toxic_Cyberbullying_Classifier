# Dataset Plan

Recommended source mix:

1. Multi-Labeled Bengali Toxic Comments — primary multi-label source (vulgar, hate, religious, threat, troll, insult).
2. Bengali Cyber Bullying Dataset — political, sexual, troll, threat, neutral.
3. BIDWESH — regional Bangla hate speech covering Barishal, Noakhali and Chittagong dialects with hate, type and target annotations.
4. Bangla-ToCo — context-aware toxic/non-toxic comments with news title, metadata and predecessor/successor context.

Download links:
- https://www.kaggle.com/datasets/tanveerbelaliut/multi-labeled-bengali-toxic-comments
- https://www.kaggle.com/datasets/moshiurrahmanfaisal/bangla-cyber-bullying-dataset
- https://data.mendeley.com/datasets/bpkrvf882k/1
- https://data.mendeley.com/datasets/gphbs7vsbz/3

Place the downloaded CSV files in `data/raw/`.

Important: Multi-Labeled Bengali Toxic Comments was itself constructed from other public Bengali abusive/hate/cyberbullying resources. Global exact-text deduplication is therefore enabled in the preparation script to reduce direct leakage.
