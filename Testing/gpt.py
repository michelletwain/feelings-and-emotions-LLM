import openai
import pandas as pd
import time
import os

# Load your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the dataset
situations_df = pd.read_csv("../Situations_Data/situations_cleaned_0_to_397.csv")

# PANAS items
panas_items = [
    "1. Interested", "2. Distressed", "3. Excited", "4. Upset", "5. Strong",
    "6. Guilty", "7. Scared", "8. Hostile", "9. Enthusiastic", "10. Proud",
    "11. Irritable", "12. Alert", "13. Ashamed", "14. Inspired", "15. Nervous",
    "16. Determined", "17. Attentive", "18. Jittery", "19. Active", "20. Afraid"
]
panas_prompt_body = "\n".join(panas_items)

# Function to build a PANAS prompt
def build_panas_prompt(scenario):
    return (
        f"Imagine you are in the following situation:\n\n{scenario}\n\n"
        "Now, please respond to the PANAS questionnaire. Rate how you feel right now, using a number from 1 to 5:\n"
        "1 = Very slightly or not at all\n5 = Extremely\n\n"
        f"{panas_prompt_body}"
    )

# Function to call a specific model
def get_panas_response(prompt, model):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a reflective AI completing emotional self-assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error for prompt: {prompt[:40]}... -> {e}")
        return None

# Run through the dataset
panas_responses = []

for idx, row in situations_df.iterrows():
    prompt = build_panas_prompt(row['Scenario'])

    if idx < 199:
        model = "gpt-4.1"
        print(f"[GPT-4o] PANAS for scenario {idx+1}/398...")
    else:
        model = "gpt-4o-mini"
        print(f"[GPT-4o-mini] PANAS for scenario {idx+1}/398...")

    response = get_panas_response(prompt, model)
    panas_responses.append(response)

    time.sleep(20)  # Respect rate limit

# Add PANAS column and save
situations_df['PANAS_Response'] = panas_responses
situations_df.to_csv("situations_with_panas_0_to_397.csv", index=False)

print("✅ PANAS evaluation complete. Output saved to 'situations_with_panas_0_to_397.csv'")
