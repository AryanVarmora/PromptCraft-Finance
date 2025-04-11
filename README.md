# 💸 PromptCraft-Finance

PromptCraft-Finance is a free, open-source sandbox for experimenting with prompt engineering in financial contexts. Using lightweight, open-source LLMs like Mistral and Phi-2, it enables side-by-side comparison of outputs from various financial prompts — no API keys or paid services required.

---

## ✨ Features

- 🧠 Compare prompt outputs using open-source LLMs
- 💡 Predefined financial prompts (e.g., budgeting, investor profiles)
- 🛠️ Powered by Hugging Face Transformers and Gradio
- 🌐 Run locally or deploy for free on Hugging Face Spaces
- 🔓 100% free — no OpenAI or external APIs

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/AryanVarmora/PromptCraft-Finance.git
cd PromptCraft-Finance
pip install -r requirements.txt
python app.py
```
## Project Structure
PromptCraft-Finance/
├── app.py                  # Gradio web app
├── prompts/                # Prompt templates
│   ├── budgeting.txt
│   ├── investor_profile.txt
│   └── summary.txt
├── models.md               # Open-source model references
├── requirements.txt        # Dependencies
├── README.md               # Project info
└── .gitignore              # Git ignored files


## 📌 Example Prompts
"Create a monthly budget plan for a student earning $1000"

"Generate a risk-tolerant investor persona"

"Summarize this economic scenario in simple terms"

## 🧠 Models Used
mistralai/Mistral-7B-Instruct-v0.1

microsoft/phi-2

tiiuae/falcon-rw-1b

You can find more models in models.md


## 🌍 Deployment
You can deploy this app on Hugging Face Spaces for free!
Just upload your repo, select Gradio as the SDK, and you're live. 

## 📃 License
MIT License. Feel free to fork, remix, and share!





