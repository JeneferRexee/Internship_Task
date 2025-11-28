import os
import json
import pdfplumber
import re
from datetime import datetime
from flask import Flask, request, jsonify

# ------------------ Load Config ------------------
with open("config.json", "r") as f:
    config = json.load(f)

UPLOAD_DIR = config["upload_directory"]
ICD_PATTERN = config["icd_pattern"]

# Create folders if not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ------------------ Logging ------------------
def log(message: str):
    logfile = "logs/app.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# ------------------ Extract ICD Codes + Page Info ------------------
def extract_page_data(pdf_path: str):
    try:
        pdf_data = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True) or ""
                codes = re.findall(ICD_PATTERN, text)

                words = page.extract_words()  
                word_coords = [
                    {
                        "text": w.get("text", ""),
                        "x0": round(w.get("x0", 0), 2),
                        "y0": round(w.get("top", 0), 2),
                        "x1": round(w.get("x1", 0), 2),
                        "y1": round(w.get("bottom", 0), 2)
                    }
                    for w in words
                ]

                page_dict = {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "text_sample": text[:150],
                    "extracted_codes": list(set(codes)),
                    "words": word_coords
                }

                pdf_data.append(page_dict)

        log(f"PDF processed successfully: {pdf_path}")
        return pdf_data

    except Exception as e:
        log(f"PDF processing failed: {e}")
        raise

# ------------------ Flask App ------------------
app = Flask(__name__)

@app.route("/extract", methods=["POST"])
def extract():
    try:
        data = request.json
        filename = data.get("filename")

        if not filename or not filename.lower().endswith(".pdf"):
            return jsonify({"status": "error", "message": "Please provide a valid PDF filename in 'uploads/' folder"}), 400

        pdf_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.isfile(pdf_path):
            return jsonify({"status": "error", "message": f"File not found in uploads/: {filename}"}), 400

        output = extract_page_data(pdf_path)
        return jsonify({"status": "success", "data": output})

    except Exception as e:
        log(f"API Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3333)
