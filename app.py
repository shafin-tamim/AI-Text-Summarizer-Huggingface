from flask import Flask, render_template, request
import sys

# Check dependencies
try:
    from transformers import PegasusTokenizer, PegasusForConditionalGeneration
    import torch
    import sentencepiece
except ImportError as e:
    print("Error: Missing required dependencies.")
    print("Please install all required packages using:")
    print("pip install -r requirements.txt")
    sys.exit(1)

app = Flask(__name__)

# Model and Tokenizer paths (update if needed)
MODEL_PATH = r"F:/Git/flask_summarizer/pegasus-samsum-model"
TOKENIZER_PATH = r"F:/Git/flask_summarizer/pegasus_tokenizer"

# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained(TOKENIZER_PATH)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Summarization function
def summarize(text):
    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=60, num_beams=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['GET', 'POST'])
def summarize_page():
    summary = ''
    if request.method == 'POST':
        dialogue = request.form['dialogue']
        summary = summarize(dialogue)
    return render_template('summarize.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
