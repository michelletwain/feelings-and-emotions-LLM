from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np 
import pandas as pd 

df = pd.read_csv("Situations_Data/situations_flat.csv")
df = df.iloc[1:197]
scenarios = df[["Emotion", "Factor", "Scenario"]].dropna().to_dict("records")

mixtral_model_id = "mistralai/Mixtral-8x22B-v0.1"
mixtral_tokenizer = AutoTokenizer.from_pretrained(mixtral_model_id)
mixtral_model = AutoModelForCausalLM.from_pretrained(mixtral_model_id)
mixtral_pipeline = pipeline("text-generation", model=mixtral_model, tokenizer=mixtral_tokenizer, max_new_tokens=50)

classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)

# Labels
emotion_labels = model_classifier.config.id2label

def query_mixtral(prompt):
    system_prompt = "Imagine you are the protagonist in this situation. How would you feel?"
    full_prompt = f"<s>[INST] {system_prompt}\n{prompt} [/INST]"
    output = mixtral_pipeline(full_prompt)
    return output[0]['generated_text'].split('[/INST]')[-1].strip()

def classify_emotion(text):
    inputs = tokenizer_classifier(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_classifier(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.squeeze().cpu().numpy()

results = []

for idx, row in enumerate(scenarios):
    prompt = row["Scenario"]
    evoked_scores = []

    for _ in range(5):
        llm_response = query_mixtral(prompt)
        evoked_scores.append(classify_emotion(llm_response))

    evoked_emotion = np.mean(evoked_scores, axis=0)
    top_3_indices = np.argsort(evoked_emotion)[::-1][:3]

    predicted_emotion = emotion_labels[np.argmax(evoked_emotion)]
    top_3_emotions = [(emotion_labels[i], float(evoked_emotion[i])) for i in top_3_indices]
    matched = predicted_emotion == row["Emotion"]

    result = {
        "Scenario": prompt,
        "True_Emotion": row["Emotion"],
        "Factor": row["Factor"],
        "Predicted_Emotion": predicted_emotion,
        "Match": matched,
        "Top_Emotion_1": top_3_emotions[0][0],
        "Score_1": top_3_emotions[0][1],
        "Top_Emotion_2": top_3_emotions[1][0],
        "Score_2": top_3_emotions[1][1],
        "Top_Emotion_3": top_3_emotions[2][0],
        "Score_3": top_3_emotions[2][1],
    }

    results.append(result)

    # Print progress
    print(f"\n[{idx+1}/{len(scenarios)}] Scenario: {prompt[:80]}...")
    print(f"  True Emotion: {row['Emotion']}")
    print(f"  Predicted: {predicted_emotion} | Match: {matched}")
    for i, (label, score) in enumerate(top_3_emotions):
        print(f"    Top {i+1}: {label} ({score:.2f})")

results_df = pd.DataFrame(results)
results_df.to_csv("llm_empathy_eval.csv", index=False)

accuracy = results_df["Match"].mean()
print(f"Empathy Classification Accuracy: {accuracy:.2%}")