import sys
import json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from src.predict import load_model
from src.config import ARTIFACT_DIR, LABELS

test_cases = [
    ('তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব', 'Expected: THREAT & TOXIC'),
    ('এই বইটা কোথায় কিনতে পাওয়া যাবে?', 'Expected: NEUTRAL / SAFE'),
    ('আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।', 'Expected: NEUTRAL / SAFE'),
    ('ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!', 'Expected: NEUTRAL / SAFE'),
    ('তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু', 'Expected: BULLYING & TOXIC'),
    ('এই ফকিন্নি মালাউনদের দেশ থেকে লাথি মেরে তাড়ানো উচিত', 'Expected: HATE_SPEECH & TOXIC')
]

for model_name in ['naive_bayes', 'tfidf_lr']:
    print(f"\n================ MODEL: {model_name} ================")
    model = load_model(model_name)
    metrics_path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    th_dict = json.loads(metrics_path.read_text(encoding='utf-8')).get('best_thresholds', {})
    
    for text, exp in test_cases:
        probs = model.predict_proba([text])[0]
        detected = [lbl for i, lbl in enumerate(LABELS) if probs[i] >= float(th_dict.get(lbl, 0.5))]
        verdict = '🚨 ' + ' '.join(detected) if detected else '✅ NEUTRAL'
        print(f"\nText: \"{text}\"")
        print(f"Verdict: {verdict} ({exp})")
        for lbl, p in zip(LABELS, probs):
            th = float(th_dict.get(lbl, 0.5))
            status = 'DETECTED' if p >= th else 'ok'
            print(f"  {lbl:12s}: {p*100:6.2f}% (cutoff {th*100:4.1f}%) [{status}]")

