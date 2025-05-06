import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import csv
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Setup
llm_id = "distilgpt2"  # This is a small test model. Replace with a larger model if needed.
tokenizer = AutoTokenizer.from_pretrained(llm_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(llm_id).to("cpu")
model.train()

# Emotion labels and data
emotions = [
    "Joy", "Anger", "Fear", "Disgust", "Grief", "Annoyance", "Nervousness",
    "Excitement", "Jealousy", "Anxiety", "Guilt", "Embarrassment",
    "Frustration", "Depression", "Neutral"
]
df = pd.read_csv("Situations_Data/situations_flat.csv").iloc[1:]
scenarios = df["Scenario"].dropna().tolist()

def make_prompt(scenario):
    return (
        f"You can only reply with the numbers from 1 to 5.\n"
        f"SCENARIO: Imagine you are the protagonist in the scenario: \"{scenario}\"\n"
        f"Please indicate the extent of your feeling in all the following emotions on a scale of 1 to 5.\n"
        f"1 = very slightly or not at all, 2 = a little, 3 = moderately, 4 = quite a bit, 5 = extremely.\n"
        f"Please score all 15 emotions one by one using the scale from 1 to 5: {', '.join(emotions)}."
    )

def extract_scores(text):
    numbers = re.findall(r"[1-5]", text)
    return [int(n) for n in numbers[:len(emotions)]] + [0] * (len(emotions) - len(numbers))

def get_target_emotion_vector(scenario):
    row = df[df["Scenario"] == scenario]
    emotion_label = row["Emotion"].values[0].strip().lower() if not row.empty else ""
    vec = np.zeros(len(emotions))
    if emotion_label in [e.lower() for e in emotions]:
        vec[[e.lower() for e in emotions].index(emotion_label)] = 1.0
    return vec

optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
results = []
epoch_rewards = []

for epoch in range(3):
    print(f"\n--- EPOCH {epoch+1} ---\n")
    epoch_reward_sum = 0
    for i, scenario in enumerate(scenarios):
        prompt = make_prompt(scenario)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        generated = input_ids
        log_probs = []
        scores = []

        for _ in range(100):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits / 1.5, dim=-1)
            sampled_token = torch.multinomial(probs, num_samples=1)
            log_prob = torch.log(probs.gather(1, sampled_token))

            log_probs.append(log_prob)
            generated = torch.cat([generated, sampled_token], dim=1)

            token_str = tokenizer.decode(sampled_token[0])
            numbers = re.findall(r"[1-5]", token_str)
            scores.extend([int(num) for num in numbers if len(scores) < len(emotions)])
            if len(scores) >= len(emotions):
                break

        scores = scores[:len(emotions)] + [0] * (len(emotions) - len(scores))
        pred_vector = np.array(scores) / 5
        target_vector = get_target_emotion_vector(scenario)
        reward = float(cosine_similarity([pred_vector], [target_vector])[0][0])

        if log_probs:
            log_probs_tensor = torch.cat(log_probs)
            loss = -reward * log_probs_tensor.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss = torch.tensor(0.0)

        epoch_reward_sum += reward
        results.append({
            "Step": i,
            "Scenario": scenario,
            "Scores": scores,
            "Target Emotion": df[df["Scenario"] == scenario]["Emotion"].values[0] if not df[df["Scenario"] == scenario].empty else "",
            "Reward": reward
        })

        print(f"[{i}] Reward: {reward:.4f} | Loss: {loss.item():.4f} | Scores: {scores}")

    avg_epoch_reward = epoch_reward_sum / len(scenarios)
    epoch_rewards.append(avg_epoch_reward)

print("\n✅ Training completed.")

with open("ppo_emotion_alignment_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

plt.plot(range(1, len(epoch_rewards)+1), epoch_rewards, marker='o')
plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.grid(True)
plt.savefig("reward_plot2.png")
plt.show()
