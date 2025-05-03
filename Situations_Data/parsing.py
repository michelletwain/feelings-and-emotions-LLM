import os
import json
import pandas as pd

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'situations.json')

with open(json_path, 'r', encoding='utf-8') as f:
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
df.to_csv(os.path.join(script_dir, 'situations_flat.csv'), index=False, quoting=1)  # quoting=csv.QUOTE_ALL

print(f"Extracted {len(df)} scenarios.")