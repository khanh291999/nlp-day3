import os
from flask import Flask, render_template, request
from transformers import pipeline
import torch

app = Flask(__name__)

# Optional: speed up first load by disabling progress bars
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Pick device (GPU if available, else CPU)
device = 0 if torch.cuda.is_available() else -1

# Load a lightweight, widely used English sentiment model
# Returns labels like "POSITIVE"/"NEGATIVE" and a score in [0,1]
sentiment = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        if text:
            # transformers handles tokenization & truncation internally
            output = sentiment(text, truncation=True)[0]
            label = "Positive" if output["label"].upper().startswith("POS") else "Negative"
            confidence = f"{output['score'] * 100:.2f}"
            result = {"label": label, "confidence": confidence}

    return render_template("index.html", result=result, text=text)

if __name__ == "__main__":
    # debug=True for auto-reload during development
    app.run(debug=True)
