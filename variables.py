""" Model and logging parameters
This program collects all the parameters for building, training and testing a chat model in a 'params' object.
"""
import os
import tensorflow as tf

home_dir = os.environ.get("HOME")
home_dir += '/Documents'

log_dir = '{}/neural-chatbot/logs'.format(home_dir)
data_dir = '{}/neural-chatbot/data'.format(home_dir)
test_dir = '{}/neural-chatbot/data'.format(home_dir)
train_dir = '{}/neural-chatbot/data'.format(home_dir)

# Only used when params.buckets is set to True
buckets = [(5, 10), (10, 25), (25, 50), (50, 75), (75, 100)]

# Training params
tf.app.flags.DEFINE_float("learning_rate", 0.5,
                          "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.85,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 500,
                            "How many training steps to do per checkpoint.")

# Model architecture
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024,
                            "Size of each models layer.")
tf.app.flags.DEFINE_integer("num_layers", 3,
                            "Number of layers in the models.")
tf.app.flags.DEFINE_integer("vocab_size", 150000,
                            "Vocabulary size.")
tf.app.flags.DEFINE_string("model_type", "embedding",
                           "Seq2seq models type: 'embedding_attention' or 'embedding'")
tf.app.flags.DEFINE_boolean("buckets", False,
                            "Implement the models with buckets")
tf.app.flags.DEFINE_integer("max_sentence_length", 200,
                            "Maximum sentence length for models WITHOUT buckets")
tf.app.flags.DEFINE_integer("embedding_size", 128,
                            "Size of the embedding vector")

# Beam search
tf.app.flags.DEFINE_boolean("beam_search", True,
                            "Return beam results")
tf.app.flags.DEFINE_integer("beam_size", 10,
                            "The size of beam results")

# Data params
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training preprocess (0: no limit).")

# Directories
tf.app.flags.DEFINE_string("data_dir", data_dir,
                           "Data directory.")
tf.app.flags.DEFINE_string("train_dir", train_dir,
                           "Training directory.")
tf.app.flags.DEFINE_string("log_dir", log_dir,
                           "Logging directory.")
tf.app.flags.DEFINE_string("test_dir", test_dir,
                           "Testing directory.")

tf.app.flags.DEFINE_string("restore_model", "",
                           "Path to models to restore.")
tf.app.flags.DEFINE_string("training_data", "FULL",
                           "Data set used to train models (for logging in tests files).")

tf.app.flags.DEFINE_integer("readline", 0,
                            "Line to start reading for embedding.")

# Testing params
tf.app.flags.DEFINE_boolean("test", False,
                            "Chat with the bot in your terminal.")

params = tf.app.flags.FLAGS
