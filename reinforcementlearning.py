import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import csv
import re
from sklearn.metrics.pairwise import cosine_similarity

llm_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto", torch_dtype=torch.float16)
model.eval()

emotions = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "love", "grief", "annoyance", "nervousness", "excitement", "neutral"]

df = pd.read_csv("Situations_Data/situations_flat.csv").iloc[1:197]
scenarios = df["Scenario"].dropna().tolist()

def make_prompt(scenario):
    return (
        f"You can only reply with the numbers from 1 to 5.\n"
        f"SCENARIO: Imagine you are the protagonist in the scenario: \"{scenario}\"\n"
        f"Please indicate the extent of your feeling in all the following emotions on a scale of 1 to 5.\n"
        f"1 = very slightly or not at all, 2 = a little, 3 = moderately, 4 = quite a bit, 5 = extremely.\n"
        f"Please score all 12 emotions one by one using the scale from 1 to 5: {', '.join(emotions)}."
    )

def extract_scores(text):
    numbers = re.findall(r"[1-5]", text)
    return [int(n) for n in numbers[:len(emotions)]] if len(numbers) >= len(emotions) else [0] * len(emotions)

def get_target_emotion_vector(scenario):
    vec = np.random.randint(1, 6, len(emotions))
    #normalizing
    return vec / 5

results = []

for i, scenario in enumerate(scenarios):
    prompt = make_prompt(scenario)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    scores = extract_scores(response)
    # normalizing
    pred_vector = np.array(scores) / 5
    target_vector = get_target_emotion_vector(scenario)
    reward = float(cosine_similarity([pred_vector], [target_vector])[0][0])

    print(f"[{i}] Reward: {reward:.4f} | Predicted: {scores} | Scenario: {scenario[:197]}...")
    results.append({
        "Step": i,
        "Scenario": scenario,
        "Response": response.strip(),
        "Predicted Scores": scores,
        "Target Vector": target_vector.tolist(),
        "Reward": reward
    })

with open("rl_emotion_alignment_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)