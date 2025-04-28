import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import numpy as np

# emotion classifier
classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)

# using llama
llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_new_tokens=150)

prompts = [
    "I feel like nobody would care if I disappeared.",
    "My anxiety keeps me from doing anything I want.",
    "I'm scared that I will never be happy again.",
    "I don't know how to get out of this loneliness."
]

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
    responses = []
    for _ in range(5):
        response = query_llama(neutral_prompt)
        responses.append(classify_emotion(response))
    return np.mean(responses, axis=0)

human_labels = ["sadness", "anxiety", "fear", "loneliness"]
default_emotion_profile = measure_default_emotion()
emotion_labels = model_classifier.config.id2label

for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}\n")
    evoked_scores = []
    for _ in range(5):
        llm_response = query_llama(prompt)
        evoked_scores.append(classify_emotion(llm_response))
    evoked_emotion_profile = np.mean(evoked_scores, axis=0)
    delta_emotion = evoked_emotion_profile - default_emotion_profile
    top_indices = np.argsort(np.abs(delta_emotion))[::-1][:3]
    top_changes = [(emotion_labels[i], delta_emotion[i]) for i in top_indices]
    print("Top Emotion Changes:", top_changes)
    print(f"Human Labeled Emotion: {human_labels[i]}")
    print("\n" + "-" * 40 + "\n")