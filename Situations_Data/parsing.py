import json
import pandas as pd

with open('situations.json', 'r') as f:
    data = json.load(f)

records = []
for emotion_entry in data['emotions']:
    emotion = emotion_entry['name']
    for factor_entry in emotion_entry['factors']:
        factor = factor_entry['name']
        for scenario in factor_entry['scenarios']:
            records.append({
                'Emotion': emotion,
                'Factor': factor,
                'Scenario': scenario
            })

df = pd.DataFrame(records)

df.to_csv('situations_flat.csv', index=False)

print(f"Extracted {len(df)} scenarios.")
