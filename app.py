from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

# Define los nombres de las clases
names = [
    'Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real',
    'Aratinga de Pinceles', 'Aratinga de Wagler', 'Aratinga Ojiblanca', 'Aratinga Orejigualda',
    'Aratinga Pertinaz', 'Batará Barrado', 'Batará Crestibarrado', 'Batara Crestinegro',
    'Batará Mayor', 'Batará Pizarroso Occidental', 'Batará Unicolor', 'Cacatua Ninfa',
    'Catita Frentirrufa', 'Cotorra Colinegra', 'Cotorra Pechiparda', 'Cotorrita Alipinta',
    'Cotorrita de Anteojos', 'Guacamaya Roja', 'Guacamaya Verde', 'Guacamayo Aliverde',
    'Guacamayo azuliamarillo', 'Guacamayo Severo', 'Hormiguerito Coicorita Norteño',
    'Hormiguerito Coicorita Sureño', 'Hormiguerito Flanquialbo', 'Hormiguerito Leonado',
    'Hormiguerito Plomizo', 'Hormiguero Azabache', 'Hormiguero Cantor', 'Hormiguero de Parker',
    'Hormiguero Dorsicastaño', 'Hormiguero Guardarribera Oriental', 'Hormiguero Inmaculado',
    'Hormiguero Sencillo', 'Hormiguero Ventriblanco', 'Lorito Amazonico', 'Lorito Cabecigualdo',
    'Lorito de fuertes', 'Loro Alibronceado', 'Loro Cabeciazul', 'Loro Cachetes Amarillos',
    'Loro Corona Azul', 'Loro Tumultuoso', 'Ojodefuego Occidental', 'Periquito Alas Amarillas',
    'Periquito Australiano', 'Periquito Barrado', 'Tiluchí Colilargo', 'Tiluchí de Santander',
    'Tiluchi Lomirrufo'
]

# Cargar el modelo
dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, 'model_VGG16_v4.keras')
modelt = load_model(model_path)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:3000"}})  # Ajusta el origen según sea necesario

# Configurar la carpeta donde se guardarán las imágenes subidas
UPLOAD_FOLDER = os.path.join(dirname, 'uploaded_images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Asegurarse de que la carpeta de subida exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Endpoint para el método GET
@app.route('/api/', methods=['GET'])
def get_example():
    return jsonify({"message": "Este es un ejemplo de respuesta GET"})

# Endpoint para aceptar imágenes mediante POST
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Leer y preprocesar la imagen
        imaget = cv2.imread(filepath)
        imaget = cv2.resize(imaget, (224, 224))
        xt = preprocess_input(np.array(imaget))
        xt = np.expand_dims(xt, axis=0)

        # Obtener predicciones
        preds = modelt.predict(xt)

        # Obtener la clase predicha y la confianza
        predicted_class_index = np.argmax(preds)
        predicted_class_name = names[predicted_class_index]
        confidence_percentage = preds[0][predicted_class_index] * 100

        return jsonify({
            "message": f'Clase predicha: {predicted_class_name}, Porcentaje de confianza: {confidence_percentage:.2f}%',
            "file_path": filepath
        }), 200
    
    return jsonify({"error": "Invalid file type. Only png, jpg, jpeg, gif are allowed."}), 400

# Función para verificar tipos de archivos permitidos
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Endpoint para el método POST
@app.route('/api/post_example', methods=['POST'])
def post_example():
    data = request.get_json()
    return jsonify({"message": "Datos recibidos correctamente", "data": data})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Recurso no encontrado"}), 404

@app.route('/')
def serve_interface():
    return send_from_directory('.', 'index.html')

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
