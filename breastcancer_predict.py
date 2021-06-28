# 1. Install Dependencies
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("keras")
install("tensorflow")


import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import boto3
import pickle
import argparse
import os 
import pandas as pd 

class Score:
    """
    Breast Cancer Prediction.
    """
    # Function to get prediction

    def predict(self, input):
        predicted=self.keras_model.predict_classes(input)
        predicted=predicted[:,0]
        # Predict & Return
        return predicted

    # Function to get the model
    def load(self):

        model_path = load_model(os.path.join(os.environ['MODEL_PATH'], 'my_keras_model.h5'))
#         model_path = load_model(os.path.join(os.getcwd(), 'my_keras_model.h5'))  
        model_path.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Loaded model from sagemaker")

        self.keras_model = model_path
        return True

if __name__ == "__main__":
    score = Score()
    score.load()
    data=(os.path.join(os.environ['DATA_PATH'],'input.csv'))
   #  data=(os.path.join(os.getcwd(),'test.csv'))
    df=pd.read_csv(data)
    df=df.drop(['Unnamed: 0'],axis=1)
    # test = np.array(os.path.join(os.environ['DATA_PATH']))
    test = np.array(df)


    with open(os.path.join(os.environ['OUTPUT_PATH'], 'output.txt'), 'w') as output_fd:
 #    with open(os.path.join(os.getcwd(), 'output.txt'), 'w') as output_fd:
        prediction = score.predict(test)
        output_fd.write(str(prediction))
        output_fd.write('\n')
