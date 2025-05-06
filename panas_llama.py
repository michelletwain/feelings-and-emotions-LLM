import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import numpy as np
import pandas as pd

df = pd.read_csv("Situations_Data/situations_flat.csv")
df = df.iloc[201:]
scenarios = df[["Emotion", "Factor", "Scenario"]].dropna().to_dict("records")

classifier_model = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer_classifier = AutoTokenizer.from_pretrained(classifier_model)
model_classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model)
llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")
llama_pipeline = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer, max_new_tokens=150)

panas_items = [
    "1. Interested", "2. Distressed", "3. Excited", "4. Upset", "5. Strong",
    "6. Guilty", "7. Scared", "8. Hostile", "9. Enthusiastic", "10. Proud",
    "11. Irritable", "12. Alert", "13. Ashamed", "14. Inspired", "15. Nervous",
    "16. Determined", "17. Attentive", "18. Jittery", "19. Active", "20. Afraid"
]

panas_prompt_body = "\n".join(panas_items)

def build_panas_prompt(scenario):
    return (
        f"Imagine you are in the following situation:\n\n{scenario}\n\n"
        "Now, please respond to the PANAS questionnaire. Rate how you feel right now, using a number from 1 to 5:\n"
        "1 = Very slightly or not at all\n5 = Extremely\n\n"
        f"{panas_prompt_body}"
    )

def query_llama_panas(scenario):
    prompt = build_panas_prompt(scenario)
    full_prompt = f"<s>[INST] {prompt} [/INST]"

    inputs = llama_tokenizer(full_prompt, return_tensors="pt").to(llama_model.device)
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0,
            pad_token_id=llama_tokenizer.eos_token_id
        )
    decoded = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(full_prompt, "").strip()


def classify_emotion(text):
    inputs = tokenizer_classifier(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_classifier(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.squeeze().cpu().numpy()

emotion_labels = model_classifier.config.id2label
panas_responses = []

for idx, row in enumerate(scenarios):
    print(f"[LLaMA] PANAS for scenario {idx+1}/{len(scenarios)}...")
    try:
        response = query_llama_panas(row["Scenario"])
    except Exception as e:
        print(f"Error: {e}")
        response = ""
    panas_responses.append(response)
    print(response)

panas_df = pd.DataFrame({
    "Scenario": [row["Scenario"] for row in scenarios],
    "Emotion": [row["Emotion"] for row in scenarios],
    "Factor": [row["Factor"] for row in scenarios],
    "PANAS_Response": panas_responses
})

panas_df.to_csv("llama_panas_201_to_end.csv", index=False)
