from __future__ import print_function

import os
import sys
import json
import boto3
from tqdm import tqdm
import tensorflow as tf
from subprocess import call
from variables import params

AWS_ACCESS_KEY = 'access_key'
AWS_SECRET_KEY = 'secret_key'
BUCKET_NAME = 'henri-chatbot-data'

s3 = boto3.client('s3',
                  aws_access_key_id=AWS_ACCESS_KEY,
                  aws_secret_access_key=AWS_SECRET_KEY)

VARIABLES = [
    "learning_rate",
    "global_step",
    "projW",
    "projB",
    "encoder_embedding",
    "LSTM_encoder1_weights",
    "LSTM_encoder1_biases",
    "LSTM_encoder2_weights",
    "LSTM_encoder2_biases",
    "LSTM_encoder3_weights",
    "LSTM_encoder3_biases",
    "decoder_embedding",
    "LSTM_decoder1_weights",
    "LSTM_decoder1_biases",
    "LSTM_decoder2_weights",
    "LSTM_decoder2_biases",
    "LSTM_decoder3_weights",
    "LSTM_decoder3_biases",
]


def retrieve_data():
    print('Retrieving spacy models...')
    call("sudo python -m spacy.en.download all", shell=True)

    # Create preprocess directory
    if not os.path.exists(params.data_dir):
        os.mkdir(params.data_dir)

    # Create log directory
    if not os.path.exists(params.log_dir):
        os.mkdir(params.log_dir)

    os.chdir(params.data_dir)

    files = s3.list_objects(Bucket=BUCKET_NAME)['Contents']

    print('Retrieving preprocess...')
    for fn in tqdm(files):
        obj = fn['Key']
        if not obj.endswith("/"):
            s3.download_file(BUCKET_NAME, obj, obj)
        else:
            pass


def upload_data():
    directory = params.data_dir

    print('Uploading training preprocess...')
    fn = 'training_data'
    s3.upload_file(directory + '/' + fn, BUCKET_NAME, fn)
    print('Uploading validation preprocess...')
    fn = 'validation_data'
    s3.upload_file(directory + '/' + fn, BUCKET_NAME, fn)


def save_variables():
    from models import helper
    with tf.Session() as sess:
        model = helper.init_model(sess, True)
        model_params = sess.run(tf.all_variables())
        for i in xrange(5, len(VARIABLES)):
            if "decoder_embedding" not in VARIABLES[i]:
                print(VARIABLES[i])
                to_save = model_params[i].tolist()
                with open(params.data_dir + VARIABLES[i], 'w') as test_file:
                    test_file.write(json.dumps(to_save))
                upload_variable(VARIABLES[i])


def upload_variable(variable):
    s3.upload_file(params.data_dir + variable, BUCKET_NAME, 'trainable_variables/' + variable)


if __name__ == "__main__":
    if sys.argv[1] == "--save":
        save_variables()
    elif sys.argv[1] == "--retrieve":
        retrieve_data()
    elif sys.argv[1] == '--upload':
        upload_data()
    elif sys.argv[1] == "--variables":
        save_variables()
    else:
        print("Unknown argument...")
