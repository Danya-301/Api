from flask import Flask, request, jsonify
from flask import send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('modelo_mobilenet.keras')

# Lista de nombres de aves
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

# Directorio para subir imágenes
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Guardar la imagen
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Predecir
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_name = names[predicted_class]
        
        return jsonify({'predicted_bird': predicted_name})

@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1>Sube una imagen de un ave para predecir su tipo</h1>
            <form method="post" action="/upload" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Subir">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
