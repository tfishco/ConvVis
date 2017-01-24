from __future__ import print_function
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import random
import mnist_tester
import numpy as np
import json
import network_json
import sys

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/conv", methods=['POST'])
def conv():
    index = int(request.form['val'])
    struct = json.loads(request.form['struct'])
    image = mnist.test.images[index]
    label = mnist.test.labels[index]
    data = {}
    data['label'] = np.argmax(label)
    data['convdata'] = mnist_tester.convolution(image, label)
    data['struct'], data['no_nodes'] = network_json.get_json(struct)
    return json.dumps(data)

#@app.route("/conv-test)
#def conv-test():
#    index = random.randint(0,10000)
#    image = mnist.test.images[index]
#    label = mnist.test.labels[index]
#    features = mnist_tester.convolution(image, label)
#    return render_template("index.html",label=np.argmax(label),features=features)

if __name__ == "__main__":
    app.run()
