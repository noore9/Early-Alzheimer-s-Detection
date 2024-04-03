import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

app = Flask(__name__)

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("images", filename)

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        m = int(request.form["alg"])
        acc = pd.read_csv("Accuracy.csv")

        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join("images/", fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)

        if m == 1:
            new_model = load_model(r'C:\Users\noor\Desktop\gkhbj\CODE\model\Mobilenet.h5')
            test_image = image.load_img(mypath, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            preprocessed_img = mobilenet_v2_preprocess_input(test_image)

        elif m == 2:
            new_model = load_model(r'C:\Users\noor\Desktop\gkhbj\CODE\model\cnn.h5')
            test_image = image.load_img(mypath, target_size=(224,224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            preprocessed_img = test_image  # Depending on the preprocessing required for your CNN model

        
        
        predictions = new_model.predict(preprocessed_img)
        class_labels = ['Mild', 'Moderate', 'No-AD', 'VeryMild']  # Provide your class labels
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        a = acc.iloc[m - 1, 1]

        return render_template("template.html", text=predicted_class_label, image_name=fn, a=round(a * 100, 3))

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
