# feelings-and-emotions-LLM

Final project for EECS6895: Empirical Methods for Natural Language Processing  
Columbia University, Spring 2025

## Overview

This project explores the emotional reasoning capabilities of large language models (LLMs) in psychologically grounded scenarios. We evaluate whether state-of-the-art LLMs like **GPT-4o-mini**, **Mistral-7B**, and **LLaMA-3/2** can simulate human-like emotional responses when prompted with real-world emotional situations. Our work draws on **emotion appraisal theory** and evaluates emotional alignment through both **classification accuracy** and **self-reported affect** using PANAS.

We propose a multi-pronged evaluation approach, consisting of:

- **Emotion classification** using a DistilRoBERTa-based classifier
- **Self-assessment of internal affect** using the 20-item PANAS questionnaire
- **Reinforcement learning** with PPO to fine-tune empathy in generation
- **Confidence-aware chatbot prototyping** to improve adaptive interaction

This enables a comprehensive view of how LLMs simulate emotion—externally in text and internally via affective scoring.

---

## Experiments and Methodology

### 1. Emotion Classification Evaluation

- We prompt models like LLaMA-2-7B and Mistral-7B to act as protagonists in emotionally charged scenarios.
- Their responses are passed through a pretrained **DistilRoBERTa emotion classifier** to evaluate alignment with the ground-truth labeled emotion.
- We measure **top-1 and top-3 accuracy** to assess emotional fidelity.

### 2. PANAS-Based Affective Scoring

- Models (GPT-4o-mini, Mistral-7B, and LLaMA-3-8B) are instructed to simulate emotions using the **PANAS questionnaire**.
- Each model returns a 20-dimensional emotion vector for 398 situations.
- We analyze inter-model differences using:
  - **Cosine similarity** (for shape alignment)
  - **L2 distance** (for absolute differences)
  - **Per-item $t$-tests** (for statistically significant discrepancies)

### 3. Reinforcement Learning with PPO

- We fine-tune **sshleifer/tiny-gpt2** using **Proximal Policy Optimization (PPO)**.
- The reward is proportional to the model’s rated intensity for the target emotion.
- Training shows early signs of reward-driven emotional alignment.
- The reinforcement setup logs reward trajectories and supports future fine-tuning.

### 4. Adaptive Emotion-Aware Chatbot (Prototype)

- We develop a chatbot that incorporates **classifier confidence gaps** to detect ambiguity.
- When emotion prediction is uncertain, it prompts the user for clarification before responding.
- This is a proof-of-concept for confidence-aware, emotionally sensitive conversational agents.

---

## Dataset

We use [EmotionBench](https://github.com/CUHK-ARISE/EmotionBench), a dataset of **428 real-world situations**, each labeled with one of eight psychologically validated negative emotions:

- anger, fear, guilt, embarrassment, frustration, depression, anxiety, and jealousy

The dataset is flattened into a CSV format with `Scenario`, `Emotion`, and `Factor` columns. Our system supports any dataset with:

- textual prompts (scenarios)
- corresponding categorical emotion labels

---

## Tech Stack

- **Language Models**: GPT-4o-mini (via OpenAI API), Mistral-7B, LLaMA-2-7B, LLaMA-3-8B
- **Emotion Classifier**: `j-hartmann/emotion-english-distilroberta-base`
- **RL Framework**: `trl` for Proximal Policy Optimization (PPO)
- **Visualization**: `matplotlib`, `seaborn`, `pandas`
- **Programming Language**: Python 3
- **Execution Platforms**: Google Colab, VSCode, local machines

---

## Getting Started

Before running the project, authenticate with Hugging Face:

```bash
huggingface-cli login

Paste your Hugging Face access token when prompted.

Then run the main script:

python3 main.py
```
