import openai
import pandas as pd
import time
import os

# Load your API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the full dataset
situations_df = pd.read_csv("../Situations_Data/situations_flat.csv")

# Select the next 200 scenarios (198 to 397)
situations_df = situations_df.iloc[198:398]

# Function to call GPT-4o
def get_gpt_response(prompt, model="gpt-4o"):
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

# Loop through the selected 200 scenarios
for idx, row in situations_df.iterrows():
    prompt = row['Scenario']
    print(f"Processing {198 + idx + 1}/398...")

    gpt_response = get_gpt_response(prompt)
    responses.append(gpt_response)

    # Sleep 20 seconds after EACH request to stay under 3 RPM limit
    time.sleep(20)

# Add responses to DataFrame and save
situations_df['GPT_Response'] = responses
situations_df.to_csv("situations_with_gpt4o_responses_198_to_397.csv", index=False)

print("✅ Done! Responses 198 to 397 saved.")
