from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# Load the trained model
model = tf.keras.models.load_model(f"front.h5")

# Load or define your scaler
scaler = StandardScaler()

labels = ['Anodic', 'Cathodic', 'Corrosion']

# Load or define X_train
X_train = np.load("X_train.npy") 
# Fit the scaler on the training data
scaler.fit_transform(X_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        # Preprocess the input features
        input_features = np.array([[feature1, feature2, feature3]])

        # Normalize the input features
        normalized_input_features = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(normalized_input_features)
        predicted_class = prediction.argmax(axis=1)

    
        # Get the corresponding label
        predicted_label = labels[int(predicted_class)]

        # Return the prediction result
        return jsonify({'prediction': predicted_label})

@app.route('/aboutUs')
def aboutUs():
    return render_template("aboutUs.html")

@app.route('/contributors')
def contributors():
    return render_template("Contributors.html")


@app.route('/contactUs')
def contactUs():
    return render_template("contactUs.html")

if __name__ == '__main__':
    app.run(debug=True)

