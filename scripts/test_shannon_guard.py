import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from src.shannon_guard import train_or_load_shannon_guard, toxic_autocomplete_guard, generate_text

def test():
    print("=== Training / Loading Shannon Guard (N-gram LM) ===")
    model = train_or_load_shannon_guard(n=2)
    print(f"Total N-gram history contexts trained: {len(model)}")
    
    test_phrases = [
        "তোরে মেরে",
        "তুই একটা",
        "জবাই করে",
        "জানে শেষ",
        "ভাই আপনি"
    ]
    
    for phrase in test_phrases:
        print(f"\n--- User Typing: '{phrase}' ---")
        guard_result = toxic_autocomplete_guard(phrase, model, n=2, top_k=5)
        print("Predictions (Next Word -> Probability):")
        for word, prob in guard_result["predictions"]:
            print(f"   -> {word}: {prob*100:.2f}%")
            
        if guard_result["is_warning"]:
            print(f"🚨 {guard_result['warning_message']}")
        else:
            print("✅ কোনো তাৎক্ষণিক ক্ষতিকর শব্দের পূর্বাভাস নেই (Safe context)")
            
        completion = generate_text(model, phrase, max_words=4, n=2)
        print(f"🎲 Shannon Auto-generation: '{completion}'")

if __name__ == "__main__":
    test()
