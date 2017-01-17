from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import random
import mnist_tester
import numpy as np

app = Flask(__name__)

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tester")
def tester():
    index = random.randint(0,10000)
    image = mnist.test.images[index]
    label = mnist.test.labels[index]
    features = mnist_tester.convolution(image, label)
    return render_template("index.html",label=np.argmax(label),features=features)

if __name__ == "__main__":
    app.run()
