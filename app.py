from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="vTbOD70e1ttMKG72lDyB"
)

@app.route('/')
def home():
    return "Dori sonovchi model Flask’da ishlayapti 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Rasm topilmadi'}), 400
    
    image = request.files['image']
    image.save('input.jpg')

    result = CLIENT.infer('input.jpg', model_id="dori-sanovchi-model-wctij-wdhbo/1")
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
