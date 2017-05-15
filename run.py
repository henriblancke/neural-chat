from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from utils.preprocess import basic
from models.helper import init_model
from variables import params, buckets
from tests.test_bot import test_chatbot
# from utils import preprocess as data_utils
from tensorflow.python.platform import gfile


def read_data(data_path, max_size=None):
    """
    Helper. Reads pre-processed data from file. Puts data into requested buckets are cuts length to max_sentence_length.
    """
    data_set = [[] for _ in buckets] if params.buckets else []

    with gfile.GFile(data_path, mode="r") as data_file:

        prompt = data_file.readline()
        response = data_file.readline()

        counter = 0
        pbar = tqdm(total=max_size)
        while prompt and response and (not max_size or counter < max_size):
            # Skip empty lines
            if (len(prompt.strip().split()) > 1) and (len(response.strip().split()) > 1):

                # Update counter
                counter += 1
                # Update progress bar
                pbar.update(1)

                prompt_ids = [int(x) for x in prompt.split()]
                response_ids = [int(x) for x in response.split()]
                response_ids.append(basic.EOS_ID)

                if params.buckets:
                    for bucket_id, (prompt_size, response_size) in enumerate(buckets):
                        if len(prompt_ids) < prompt_size and len(response_ids) < response_size:
                            data_set[bucket_id].append([prompt_ids, response_ids])
                            break

                else:
                    if len(prompt_ids) <= params.max_sentence_length and \
                                    len(response_ids) <= params.max_sentence_length:
                        data_set.append([prompt_ids, response_ids])

            prompt, response = data_file.readline(), data_file.readline()

        epoch = counter / params.batch_size
        pbar.close()

    return data_set, epoch


def train():
    print("Preparing preprocess in %s" % params.data_dir)
    data_train, data_dev, _ = basic.prepare_data(params.data_dir, params.vocab_size)

    with tf.Session() as sess:
        # Create models.
        model = init_model(sess, False)
        print("Created %s models with %d layers of %d units." % (params.model_type, params.num_layers, params.size))

        print("Reading pre-processed development and training data (limit: %d)." % params.max_train_data_size)
        dev_set, _ = read_data(data_dev)
        train_set, epoch = read_data(data_train, params.max_train_data_size)

        if params.buckets:
            print("Using bucketed models.")
            train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
            train_total_size = float(sum(train_bucket_sizes))
            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in xrange(len(train_bucket_sizes))]

        loss = float(0)
        step_time = float(0)
        current_step = int(0)
        previous_losses = list()

        print("Starting the training loop...")
        while True:
            start_time = time.time()

            bucket_id = None
            if params.buckets:
                # Choose a bucket according to preprocess distribution. We pick a random number
                # in [0, 1] and use the corresponding interval in train_buckets_scale.
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in xrange(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id=bucket_id)
            summaries, _, step_loss, _ = model.step(session=sess,
                                                    encoder_inputs=encoder_inputs,
                                                    decoder_inputs=decoder_inputs,
                                                    target_weights=target_weights,
                                                    bucket_id=bucket_id,
                                                    forward_only=False,
                                                    beam_search=False)

            current_step += 1
            loss += step_loss / params.steps_per_checkpoint
            step_time += (time.time() - start_time) / params.steps_per_checkpoint

            # Save checkpoint, print statistics, and run evaluations.
            if current_step % params.steps_per_checkpoint == 0:

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("%s: Global step %d learning rate: %.4f, step-time: %.2f, perplexity: %.2f" %
                      (str(datetime.now())[:-10], model.global_step.eval(),
                       model.learning_rate.eval(), step_time, perplexity))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    print("Adjusting learning rate.")
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                print("Saving checkpoint")
                checkpoint_path = os.path.join(params.train_dir, "ckpt_vocab{}_size{}.ckpt".format(params.vocab_size,
                                                                                                   params.size))
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                # Reset step time and loss
                loss = float(0)
                step_time = float(0)

                if params.buckets:
                    for bucket_id in xrange(len(buckets) - 1):
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                        _, _, eval_loss, _ = model.step(session=sess,
                                                        encoder_inputs=encoder_inputs,
                                                        decoder_inputs=decoder_inputs,
                                                        target_weights=target_weights,
                                                        bucket_id=bucket_id,
                                                        forward_only=True,
                                                        beam_search=False)

                        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                        print("Evaluation: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


def main(_):
    if params.test:
        test_chatbot()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
