
import ast
import pandas as pd

def clean_and_flat_topics(topic_str):
    if not isinstance(topic_str, str) or pd.isna(topic_str):
        return []
    
    try:
        # safely evaluate the string literal
        parsed = ast.literal_eval(topic_str)
    except (ValueError, SyntaxError):
        return []

    # Flatten logic
    flat_topics = []
    
    def flatten(item):
        if isinstance(item, list):
            for sub in item:
                flatten(sub)
        elif isinstance(item, str):
            flat_topics.append(item)
            
    flatten(parsed)
    
    # Cleaning: lowercase, strip, unique
    cleaned = sorted(list(set([t.lower().strip() for t in flat_topics if t])))
    return cleaned

# Test cases from the data inspection
train_samples = [
    "[['Ndume App', 'Poultry Feeding']]",
    "[['Poultry Housing'], ['Poultry Housing']]",
    "[['Asili Fertilizer (Organic)', 'Biosecurity In...']]" # truncated but structure is key
]

prior_samples = [
    "['Ndume App', 'Poultry Feeding']",
    "['Poultry Products']"
]

print("--- Testing Train Samples ---")
for s in train_samples:
    print(f"Original: {s}")
    print(f"Parsed: {clean_and_flat_topics(s)}")

print("\n--- Testing Prior Samples ---")
for s in prior_samples:
    print(f"Original: {s}")
    print(f"Parsed: {clean_and_flat_topics(s)}")
