from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo entrenado
model = load_model('model/shoe_brand_model.keras')

# Cargar las etiquetas de clase
with open('model/class_labels.json', 'r') as f:
    class_labels = json.load(f)

print("Class Labels:", class_labels)  # Verificar el contenido de class_labels

# Invertir el diccionario para que los índices sean las claves
class_indices = {int(v): k for k, v in class_labels.items()}  # Convertir índices a enteros
print("Class Indices:", class_indices)  # Verificar el contenido de class_indices

@app.route('/')
def home():
    return "Welcome to the Shoe Brand Classification API!"

@app.route('/test', methods=['GET'])
def test():
    return "Test route is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = img.resize((150, 150))  # Ajusta el tamaño a (150, 150)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalización para tu modelo
        
        # Realizar la predicción
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        
        # Asegúrate de que class_index sea un entero
        class_index = int(class_index)
        
        # Obtener el nombre de la clase
        predicted_label = class_indices.get(class_index, 'Unknown')  # Obtener el nombre de la clase
        confidence = float(prediction[0][class_index])
        
        print(f'Class Index: {class_index}, Predicted Label: {predicted_label}, Confidence: {confidence}')
        
        return jsonify({'predicted_label': predicted_label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
