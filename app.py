from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="vTbOD70e1ttMKG72lDyB"
)

MODEL_ID = "dori-sanovchi-model-wctij-wdhbo/1"

@app.route("/")
def home():
    return jsonify({"message": "✅ Dori sanovchi API ishlayapti!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Rasm yuboring (form-data orqali 'image' nomi bilan)"}), 400

    image = request.files["image"]
    image.save("temp.jpg")

    result = CLIENT.infer("temp.jpg", model_id=MODEL_ID)
    return jsonify({
        "dorilar_soni": len(result["predictions"]),
        "bashorat": result["predictions"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
