import openai
import pandas as pd
import time

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your parsed situations
situations_df = pd.read_csv("../Situations_Data/situations_flat.csv") 

# Only take the first 198 rows
situations_df = situations_df.iloc[:198]

# Function to call GPT-4o (for each scenario)
def get_gpt_response(prompt, model="gpt-4.1"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an empathetic counselor. Please respond empathetically to the user's situation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error for prompt: {prompt[:30]}... -> {e}")
        return None

responses = []

# Loop through the first 198 scenarios
for idx, row in situations_df.iterrows():
    prompt = row['Scenario']
    print(f"Processing {idx+1}/{len(situations_df)}...")

    gpt_response = get_gpt_response(prompt)
    responses.append(gpt_response)

    # Sleep 20 seconds after EACH request to stay under 3 RPM limit
    time.sleep(20)

# Save the responses
situations_df['GPT_Response'] = responses
situations_df.to_csv("situations_with_gpt4o_responses_first198.csv", index=False)

print("✅ Done! All 198 responses saved.")
