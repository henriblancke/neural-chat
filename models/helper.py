from __future__ import print_function

import variables
import tensorflow as tf
from variables import params
from .model import Seq2SeqModel
from tensorflow.python.platform import gfile


def init_model(session, forward_only):
    buckets_or_sentence_length = variables.buckets if params.buckets \
        else params.max_sentence_length
    # Beam search can only be in feed forward mode
    beam_search = params.beam_search
    if not forward_only:
        beam_search = False

    print("Initializing models...")
    model = Seq2SeqModel(vocab_size=params.vocab_size,
                         embedding_size=params.embedding_size,
                         buckets_or_sentence_length=buckets_or_sentence_length,
                         size=params.size,
                         num_layers=params.num_layers,
                         max_gradient_norm=params.max_gradient_norm,
                         batch_size=params.batch_size,
                         learning_rate=params.learning_rate,
                         learning_rate_decay_factor=params.learning_rate_decay_factor,
                         model_type=params.model_type,
                         forward_only=forward_only,
                         beam_search=beam_search,
                         beam_size=params.beam_size)

    checkpoint = tf.train.get_checkpoint_state(params.train_dir)

    if params.restore_model:
        print("Reading models parameters from: %s" % params.restore_model)
        model.saver.restore(session, params.restore_model)
    elif checkpoint and gfile.Exists(params.train_dir):
        print("Reading models parameters from: %s" % checkpoint)
        # tf.train.SummaryWriter(params.log_dir, session.graph)
        model.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        print("Created models with fresh parameters.")
        # tf.train.SummaryWriter(params.log_dir, session.graph)
        session.run(tf.global_variables_initializer())

        # TODO: Add pre-trained embeddings

    return model
