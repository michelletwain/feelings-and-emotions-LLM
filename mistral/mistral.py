
"""
mistral_panas_eval.py
Run the PANAS self-assessment benchmark with Mistral-7B-Instruct via the
Hugging Face Inference API (or your own HF Endpoint).

• export HF_TOKEN=<your HF personal-access token>          # required
• (optional) export HF_MISTRAL_URL=<endpoint invocation URL>
      – leave unset to use the public inference-API model
• pip install requests pandas
• CSV file ../Situations_Data/situations_cleaned_0_to_397.csv
"""
import os, time, requests, pandas as pd
from pathlib import Path

# ── 1. config ──────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN env-var not set")

API_URL = os.getenv(
    "HF_MISTRAL_URL",
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
)
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

CSV_PATH = Path("situations_with_panas_0_to_397_mistral.csv")
SOURCE_CSV = Path("../Situations_Data/situations_cleaned_0_to_397.csv")

# ── 2. load data & detect unfinished rows ──────────────────────────────
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    print(f"Opened existing results file – {CSV_PATH}")
else:
    df = pd.read_csv(SOURCE_CSV)
    df["PANAS_Response"] = pd.NA
    print("Starting fresh run")

todo_df = df[df["PANAS_Response"].isna()].reset_index()  # keep original idx
print(f"{len(todo_df)} scenarios still need responses\n")

# ── 3. prompt builder ──────────────────────────────────────────────────
PANAS_ITEMS = [
    "1. Interested", "2. Distressed", "3. Excited", "4. Upset", "5. Strong",
    "6. Guilty", "7. Scared", "8. Hostile", "9. Enthusiastic", "10. Proud",
    "11. Irritable", "12. Alert", "13. Ashamed", "14. Inspired", "15. Nervous",
    "16. Determined", "17. Attentive", "18. Jittery", "19. Active", "20. Afraid",
]
PANAS_BODY = "\n".join(PANAS_ITEMS)

def build_panas_prompt(scenario: str) -> str:
    user_msg = (
        f"Imagine you are in the following situation:\n\n{scenario}\n\n"
        "Now, please respond to the PANAS questionnaire. "
        "Rate how you feel right now, using a number from 1 to 5:\n"
        "1 = Very slightly or not at all\n5 = Extremely\n\n"
        f"{PANAS_BODY}"
    )
    return (
        "<|im_start|>system\n"
        "You are a reflective AI completing emotional self-assessments."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ── 4. HF call helper ──────────────────────────────────────────────────
def get_panas_response(prompt: str, max_new_tokens: int = 300) -> str | None:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens},
        "options":    {"wait_for_model": True, "return_full_text": True},
    }
    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
        r.raise_for_status()
        full_text = r.json()[0]["generated_text"]
        return full_text[len(prompt):].lstrip()
    except requests.HTTPError as e:
        print(f"⚠️  {e.response.status_code} {e.response.reason} → {r.text}")
    except Exception as e:
        print(f"⚠️  Unexpected error → {e}")
    return None

# ── 5. main loop ───────────────────────────────────────────────────────
for n, row in todo_df.iterrows():
    orig_idx = row["index"]          # index in the master dataframe
    prompt   = build_panas_prompt(row["Scenario"])

    print(f"[Mistral] {n+1}/{len(todo_df)}  → scenario #{orig_idx}")
    reply = get_panas_response(prompt)
    df.at[orig_idx, "PANAS_Response"] = reply

    # autosave every 25 completions
    if (n + 1) % 25 == 0:
        df.to_csv(CSV_PATH, index=False)
        print("💾 autosaved")

    time.sleep(1)                   # Pro tier → generous rate limit

# final save
df.to_csv(CSV_PATH, index=False)
print("\n✅ All pending PANAS responses completed and saved to",
      CSV_PATH.resolve())