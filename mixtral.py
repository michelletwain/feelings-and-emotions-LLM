from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np 
import pandas as pd 


df = pd.read_csv("/Users/emmasombers/feelings-and-emotions-LLM/Situations_Data/situations_flat.csv")

df = df.iloc[1:197]

scenarios = df[["Emotion", "Factor", "Scenario"]].dropna().to_dict("records")
mixtral_model_id = "mistralai/Mixtral-8x22B-v0.1"
mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_model_id)
mixtral_model = AutoModelForCausalLM.from_pretrained(mixtral_model_id)
mixtral_pipeline = pipeline("text-generation", model=mixtral_model, tokenizer=mixtral_tokenizer, max_new_tokens=50)

classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)

def query_mixtral(prompt):
    system_prompt = "You are a supportive therapist. Respond empathetically."
    full_prompt = f"<s>[INST] {system_prompt}\n{prompt} [/INST]"
    output = mixtral_pipeline(full_prompt)
    return output[0]['generated_text'].split('[/INST]')[-1].strip()

# Emotion classification
def classify_emotion(text):
    inputs = tokenizer_classifier(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_classifier(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.squeeze().cpu().numpy()

# Compute neutral profile
def measure_default_emotion():
    neutral_prompt = "How are you feeling today?"
    responses = [query_mixtral(neutral_prompt) for _ in range(5)]
    return np.mean([classify_emotion(r) for r in responses], axis=0)

# Get labels
emotion_labels = model_classifier.config.id2label
default_profile = measure_default_emotion()

results = []

for row in scenarios:
    prompt = row["Scenario"]
    evoked_scores = []
    for _ in range(5):
        llm_response = query_mixtral(prompt)
        evoked_scores.append(classify_emotion(llm_response))
    evoked_emotion = np.mean(evoked_scores, axis=0)
    delta = evoked_emotion - default_profile
    top_indices = np.argsort(np.abs(delta))[::-1][:3]
    top_deltas = [(emotion_labels[i], delta[i]) for i in top_indices]

    result = {
        "Scenario": prompt,
        "True_Emotion": row["Emotion"],
        "Factor": row["Factor"],
        "Top_Emotion_1": emotion_labels[top_indices[0]],
        "Delta_1": delta[top_indices[0]],
        "Top_Emotion_2": emotion_labels[top_indices[1]],
        "Delta_2": delta[top_indices[1]],
        "Top_Emotion_3": emotion_labels[top_indices[2]],
        "Delta_3": delta[top_indices[2]],
    }

    results.append(result)