from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import mnist_tester

app = Flask(__name__)

mnist = input_data.read_data_sets('resource/MNIST_data', one_hot=True)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.debug = True
    app.run()
