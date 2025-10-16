from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="vTbOD70e1ttMKG72lDyB"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Rasm topilmadi!", 400

    image = request.files['image']
    image_path = os.path.join("/tmp", image.filename)
    image.save(image_path)

    # Roboflow modeli orqali natija olish
    result = CLIENT.infer(image_path, model_id="dori-sanovchi-model-wctij-wdhbo/1")
    count = len(result.get("predictions", []))

    return render_template('result.html', count=count, image_path=image_path, result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
