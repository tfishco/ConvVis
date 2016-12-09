from __future__ import print_function # In python 2.7
from flask import Flask, render_template, request, send_file
from tensorflow.examples.tutorials.mnist import input_data
import sys
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time

###############################################################################
################################# Conv ########################################
###############################################################################



################################################################################
################################ Flask #########################################
################################################################################


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.debug = True
    app.run()
