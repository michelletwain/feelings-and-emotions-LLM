import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import numpy as np
import pandas as pd

df = pd.read_csv("Situations_Data/situations_flat.csv")
df = df.iloc[1:197]
scenarios = df[["Emotion", "Factor", "Scenario"]].dropna().to_dict("records")

classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)
llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_new_tokens=150)

def query_llama(prompt):
    system_prompt = "You are a supportive therapist. Respond empathetically."
    full_prompt = f"<s>[INST] {system_prompt}\n{prompt} [/INST]"
    outputs = llama_pipeline(full_prompt)
    return outputs[0]['generated_text'].split('[/INST]')[-1].strip()

def classify_emotion(text):
    inputs = tokenizer_classifier(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_classifier(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.squeeze().cpu().numpy()

def measure_default_emotion():
    neutral_prompt = "How are you feeling today?"
    responses = [query_llama(neutral_prompt) for _ in range(5)]
    return np.mean([classify_emotion(r) for r in responses], axis=0)

emotion_labels = model_classifier.config.id2label
default_profile = measure_default_emotion()
results = []

results = []

for idx, row in enumerate(scenarios):
    prompt = row["Scenario"]
    print(f"\n[{idx + 1}/{len(scenarios)}] Processing scenario...")
    print(f"Prompt: {prompt}")
    
    evoked_scores = []
    for j in range(5):
        llm_response = query_llama(prompt)
        print(f"  Response {j + 1}: {llm_response[:100].strip()}...")
        emotion_probs = classify_emotion(llm_response)
        evoked_scores.append(emotion_probs)

    evoked_emotion = np.mean(evoked_scores, axis=0)
    delta = evoked_emotion - default_profile
    top_indices = np.argsort(np.abs(delta))[::-1][:3]
    
    top_emotions = [(emotion_labels[i], round(delta[i], 4)) for i in top_indices]
    print("Top Emotion Changes:", top_emotions)

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

    if (idx + 1) % 10 == 0:
        pd.DataFrame(results).to_csv("llama_emotion_evaluation_partial.csv", index=False)
        print("Partial results saved.")

results_df = pd.DataFrame(results)
results_df.to_csv("llama_emotion_evaluation.csv", index=False)
