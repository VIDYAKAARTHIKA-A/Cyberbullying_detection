from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained("cyberbully-bert")
model = BertForSequenceClassification.from_pretrained("cyberbully-bert")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return jsonify({
        "text": text,
        "label": "cyberbullying" if prediction == 1 else "not cyberbullying"
    })

if __name__ == "__main__":
    app.run(debug=True)
