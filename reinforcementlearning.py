import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd
import numpy as np
import csv
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from random import shuffle

classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)
model_classifier.eval()

llm_id = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(llm_id).to("cpu")
model.train()

emotions = ["Joy", "Anger", "Pride", "Fear", "Excitement", "Jealousy", "Anxiety", "Guilt", "Embarrassment", "Frustration", "Depression", "Neutral"]
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
    return [int(n) for n in numbers[:len(emotions)]] if len(numbers) >= len(emotions) else [0] * len(emotions)

def get_target_emotion_vector(scenario):
    row = df[df["Scenario"] == scenario]
    if row.empty:
        return np.zeros(len(emotions))
    emotion_label = row["Emotion"].values[0].strip().lower()
    vec = np.zeros(len(emotions))
    if emotion_label in [e.lower() for e in emotions]:
        idx = [e.lower() for e in emotions].index(emotion_label)
        vec[idx] = 1.0
    return vec

#PPO
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
results = []
epoch_rewards = []

for epoch in range(5):
    print(f"\n--- EPOCH {epoch+1} ---\n")
    shuffle(scenarios)
    epoch_reward_sum = 0
    for i, scenario in enumerate(scenarios):
        prompt = make_prompt(scenario)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        generated = input_ids
        log_probs = []
        sampled_tokens = []
        scores = []

        for _ in range(100):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits / 1.5, dim=-1)
            sampled_token = torch.multinomial(probs, num_samples=1)
            log_prob = torch.log(probs.gather(1, sampled_token))

            log_probs.append(log_prob)
            sampled_tokens.append(sampled_token)
            generated = torch.cat([generated, sampled_token], dim=1)

            token_str = tokenizer.decode(sampled_token[0])
            numbers = re.findall(r"[1-5]", token_str)
            if numbers:
                for num in numbers:
                    if len(scores) < len(emotions):
                        scores.append(int(num))
            if len(scores) >= len(emotions):
                break

        scores = scores[:len(emotions)] + [0] * (len(emotions) - len(scores))
        scores = scores[:len(emotions)] + [0] * (len(emotions) - len(scores))
        target_vector = get_target_emotion_vector(scenario)
        target_index = np.argmax(target_vector)
        reward = scores[target_index] / 5.0

        clipped_reward = max(min(reward, 1.0), -1.0)
        if len(log_probs) > 0:
            log_probs_tensor = torch.cat(log_probs)
            loss = -clipped_reward * log_probs_tensor.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss = torch.tensor(0.0)

        print(f"[{i}] Reward: {reward:.4f} | Loss: {loss.item():.4f} | Scores: {scores}")

        epoch_reward_sum += reward
        results.append({
            "Step": i,
            "Scenario": scenario,
            "Scores": scores,
            "Target Emotion": df[df["Scenario"] == scenario]["Emotion"].values[0] if not df[df["Scenario"] == scenario].empty else "",
            "Reward": reward
        })

    avg_epoch_reward = epoch_reward_sum / len(scenarios)
    epoch_rewards.append(avg_epoch_reward)

print("\nTraining completed.")

with open("ppo_emotion_alignment_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

plt.plot(range(1, len(epoch_rewards)+1), epoch_rewards, marker='o', linestyle='')
plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.grid(True)
plt.savefig("reward_plot.png")
plt.show()