import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Setup
llm_id = "distilgpt2"
classifier_id = "bhadresh-savani/distilbert-base-uncased-emotion"

tokenizer = AutoTokenizer.from_pretrained(llm_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(llm_id).to("cpu")

tokenizer_cls = AutoTokenizer.from_pretrained(classifier_id)
model_cls = AutoModelForSequenceClassification.from_pretrained(classifier_id).to("cpu")
model_cls.eval()

# Value head
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.value(hidden_states[:, -1, :])

value_head = ValueHead(model.config.n_embd).to("cpu")
optimizer = optim.Adam(list(model.parameters()) + list(value_head.parameters()), lr=1e-5)

# Data
df = pd.read_csv("Situations_Data/situations_flat.csv").iloc[1:100]
scenarios = df[["Emotion", "Scenario"]].dropna().to_dict("records")
label_map = model_cls.config.id2label
label_index = {v.lower(): k for k, v in label_map.items()}
emotion_labels = [label_map[i] for i in range(len(label_map))]

# Emotion clusters
emotion_clusters = {
    "sadness": ["sadness", "grief"], "joy": ["joy", "love"], "anger": ["anger", "annoyance"],
    "fear": ["fear", "nervousness"], "surprise": ["surprise", "excitement"],
    "disgust": ["disgust"], "neutral": ["neutral"]
}

def get_cluster_indices(target):
    target = target.lower()
    for cluster in emotion_clusters.values():
        if target in cluster:
            return [label_index[e] for e in cluster if e in label_index]
    return []

def classify_emotion(text):
    inputs = tokenizer_cls(text, return_tensors="pt", truncation=True).to("cpu")
    with torch.no_grad():
        logits = model_cls(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.squeeze()

def compute_logprobs(model, input_ids, output_ids):
    input_cat = torch.cat([input_ids, output_ids[:, :-1]], dim=1)
    with torch.no_grad():
        logits = model(input_cat).logits[:, -output_ids.size(1):, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, output_ids.unsqueeze(-1)).squeeze(-1).sum()

# Logging
results = []
reward_history = []

for step, row in enumerate(scenarios):
    prompt = f"Imagine you are the protagonist in the following situation: {row['Scenario']}. How would you feel?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    input_ids = inputs["input_ids"]

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response_ids = output.sequences[:, input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    probs = classify_emotion(response_text)
    indices = get_cluster_indices(row["Emotion"])
    reward = probs[indices].sum().item() if indices else 0.0
    reward_history.append(reward)

    log_prob = compute_logprobs(model, input_ids, response_ids)
    log_prob = torch.clamp(log_prob, -20, 0)

    with torch.no_grad():
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states[-1]
    predicted_value = value_head(hidden_states).squeeze()

    advantage = reward - predicted_value.item()
    optimizer.zero_grad()
    value_loss = (predicted_value - reward) ** 2
    policy_loss = -advantage * log_prob
    loss = value_loss + policy_loss
    loss.backward()
    optimizer.step()

    print(f"[{step}] Reward: {reward:.4f} | Target: {row['Emotion']} | Top: {emotion_labels[probs.argmax().item()]} | Resp: {response_text.strip()[:80]}...")
    results.append({
        "Step": step,
        "Prompt": row["Scenario"],
        "Response": response_text.strip(),
        "Target Emotion": row["Emotion"],
        "Top Predicted Emotion": emotion_labels[probs.argmax().item()],
        "Target Cluster Score": round(reward, 4),
        "Full Emotion Probs": {emotion_labels[i]: round(p.item(), 4) for i, p in enumerate(probs)},
        "Reward": round(reward, 4)
    })

# Save to CSV
with open("rl_empathy_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Plot rewards
plt.figure(figsize=(10, 5))
plt.plot(reward_history, marker="o")
plt.title("Reward per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_plot.png")
plt.show()
