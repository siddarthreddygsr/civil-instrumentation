from flask import Flask, request, render_template, send_from_directory
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
from keras.models import load_model
import io

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model from the H5 file
model = load_model('trained_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the image is received
        if 'image' not in request.files:
            error = "No image file provided"
            return render_template('index.html', error=error)

        img_file = request.files['image']
        if img_file:
            # Convert the image file data to an io.BytesIO object
            img_data = io.BytesIO(img_file.read())

            # Correctly handle the uploaded image
            img = image.load_img(img_data, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Make predictions on the image
            predictions = model.predict(x)

            # Depending on your model's output format, you may need to interpret the predictions accordingly
            # For example, if it's a binary classification model, you can check the class label like this:
            class_label = "There is a visble Crack" if predictions[0][0] > 0.5 else "There is no crack"

            return render_template('index.html', prediction=class_label)
    except Exception as e:
        error = str(e)
        return render_template('index.html', error=error)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5555)
