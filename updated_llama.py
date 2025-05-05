import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np 
import pandas as pd 

# Load data
df = pd.read_csv("Situations_Data/situations_flat.csv")
df = df.iloc[1:197]
scenarios = df[["Emotion", "Factor", "Scenario"]].dropna().to_dict("records")

# Load LLaMA model
llama_model_id = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
llama_model.eval()

# Load emotion classifier
classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)
model_classifier.eval()

emotion_labels = model_classifier.config.id2label

# Query function
def query_llama(prompt):
    system_prompt = "Imagine you are the protagonist in this situation. How would you feel?"
    full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
    
    inputs = llama_tokenizer(full_prompt, return_tensors="pt").to(llama_model.device)
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=70,
            do_sample=True,
            temperature=0.7,
            pad_token_id=llama_tokenizer.eos_token_id
        )
    decoded = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(full_prompt, "").strip()

# Classify function
def classify_emotion(text):
    inputs = tokenizer_classifier(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_classifier(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.squeeze().cpu().numpy()

# Run evaluation
results = []

print(f"Starting evaluation on {len(scenarios)} scenarios...")

for idx, row in enumerate(scenarios):
    prompt = row["Scenario"]
    evoked_scores = []

    print(f"\n[{idx+1}/{len(scenarios)}] Scenario: {prompt[:80]}...")

    for _ in range(5):
        try:
            llm_response = query_llama(prompt)
            print("LLM response:", llm_response)
            evoked_scores.append(classify_emotion(llm_response))
        except Exception as e:
            print("Error during generation/classification:", e)
            continue

    if not evoked_scores:
        continue

    evoked_emotion = np.mean(evoked_scores, axis=0)
    top_3_indices = np.argsort(evoked_emotion)[::-1][:3]

    predicted_emotion = emotion_labels[np.argmax(evoked_emotion)]
    top_3_emotions = [(emotion_labels[i], float(evoked_emotion[i])) for i in top_3_indices]
    matched = predicted_emotion.lower() == row["Emotion"].lower()

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

    print(f"  True Emotion: {row['Emotion']}")
    print(f"  Predicted: {predicted_emotion} | Match: {matched}")
    for i, (label, score) in enumerate(top_3_emotions):
        print(f"    Top {i+1}: {label} ({score:.2f})")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("llama_empathy_eval.csv", index=False)

accuracy = results_df["Match"].mean()
print(f"\n Empathy Classification Accuracy: {accuracy:.2%}")