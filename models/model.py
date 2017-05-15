from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from models.custom import seq2seq
from utils.preprocess import basic


class Seq2SeqModel(object):
    def __init__(self, vocab_size, embedding_size, buckets_or_sentence_length, size, num_layers,
                 max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, model_type,
                 use_lstm=True,
                 num_samples=512,
                 forward_only=False,
                 beam_search=True,
                 beam_size=10):
        """Create the models.  This constructor can be used to created an embedded or embedded-attention,
            bucketed or non-bucketed models made of single or multi-layer RNN cells.

        Args:
          vocab_size: Size of the vocabulary.
          buckets_or_sentence_length:
            If using buckets:
              A list of pairs (I, O), where I specifies maximum input length
              that will be processed in that bucket, and O specifies maximum output
              length. Training instances that have inputs longer than I or outputs
              longer than O will be pushed to the next bucket and padded accordingly.
              We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
            Else:
              Number of the maximum number of words per sentence.
          size: Number of units in each layer of the models.
          num_layers: Number of layers in the models.
          max_gradient_norm: Gradients will be clipped to maximally this norm.
          batch_size: The size of the batches used during training;
            the models construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: Learning rate to start with.
          learning_rate_decay_factor: Decay learning rate by this much when needed.
          num_samples: Number of samples for sampled softmax.
          forward_only: If set, we do not construct the backward pass in the models.
        """

        # Need to determine if we're using buckets or not:
        self.buckets = None
        if type(buckets_or_sentence_length) == list:
            self.buckets = buckets_or_sentence_length
        else:
            self.max_sentence_length = buckets_or_sentence_length

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < num_samples < self.vocab_size:
            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [size, self.vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, vocab_size)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.contrib.rnn.GRUCell(size)
        if use_lstm:
            single_cell = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True)

        # Dropout
        # TODO: Make drop-out placeholder
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=.8)

        cell = single_cell
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)

        # The seq2seq function: we use embedding for the input and attention (if applicable).
        if model_type is "embedding_attention":
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return seq2seq.embedding_attention_seq2seq(encoder_inputs=encoder_inputs,
                                                           decoder_inputs=decoder_inputs,
                                                           cell=cell,
                                                           num_encoder_symbols=self.vocab_size,
                                                           num_decoder_symbols=self.vocab_size,
                                                           embedding_size=embedding_size,
                                                           output_projection=output_projection,
                                                           feed_previous=do_decode,
                                                           beam_search=beam_search,
                                                           beam_size=beam_size)

        else:  # just build embedding models, I should probably change this to throw an error
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return seq2seq.embedding_rnn_seq2seq(encoder_inputs=encoder_inputs,
                                                     decoder_inputs=decoder_inputs,
                                                     cell=cell,
                                                     num_encoder_symbols=self.vocab_size,
                                                     num_decoder_symbols=self.vocab_size,
                                                     embedding_size=embedding_size,
                                                     output_projection=output_projection,
                                                     feed_previous=do_decode,
                                                     beam_search=beam_search,
                                                     beam_size=beam_size)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        try:
            encoder_range = self.buckets[-1][0]
            decoder_range = self.buckets[-1][1]
        except (AttributeError, TypeError):
            encoder_range = self.max_sentence_length
            decoder_range = self.max_sentence_length

        # Last bucket is the biggest one.
        for i in xrange(encoder_range):
            self.encoder_inputs.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name="encoder{0}".format(i)))

        for i in xrange(decoder_range + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,
                                                      shape=[None],
                                                      name="decoder{0}".format(i)))

            self.target_weights.append(tf.placeholder(tf.float32,
                                                      shape=[None],
                                                      name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        if self.buckets:
            if forward_only:
                if beam_search:
                    self.outputs, self.states = seq2seq.decode_model_with_buckets(
                        encoder_inputs=self.encoder_inputs,
                        decoder_inputs=self.decoder_inputs,
                        targets=targets,
                        weights=self.target_weights,
                        buckets=self.buckets,
                        seq2seq=lambda x, y: seq2seq_f(x, y, True),
                        softmax_loss_function=softmax_loss_function)
                else:
                    self.outputs, self.losses = seq2seq.model_with_buckets(encoder_inputs=self.encoder_inputs,
                                                                           decoder_inputs=self.decoder_inputs,
                                                                           targets=targets,
                                                                           weights=self.target_weights,
                                                                           buckets=self.buckets,
                                                                           seq2seq=lambda x, y: seq2seq_f(x, y, True),
                                                                           softmax_loss_function=softmax_loss_function)

                    # If we use output projection, we need to project outputs for decoding.
                    if output_projection is not None:
                        for b in xrange(len(self.buckets)):
                            self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                                               for output in self.outputs[b]]
            else:
                self.outputs, self.losses = seq2seq.model_with_buckets(encoder_inputs=self.encoder_inputs,
                                                                       decoder_inputs=self.decoder_inputs,
                                                                       targets=targets,
                                                                       weights=self.target_weights,
                                                                       buckets=self.buckets,
                                                                       seq2seq=lambda x, y: seq2seq_f(x, y, False),
                                                                       softmax_loss_function=softmax_loss_function)

        else:
            if forward_only:
                self.outputs, self.states = seq2seq_f(encoder_inputs=self.encoder_inputs,
                                                      decoder_inputs=self.decoder_inputs[:-1],
                                                      do_decode=True)

                if not beam_search:
                    self.losses = seq2seq.sequence_loss(logits=self.outputs,
                                                        targets=targets,
                                                        weights=self.target_weights[:-1],
                                                        softmax_loss_function=softmax_loss_function)

                    # Project outputs for decoding
                    if output_projection is not None:
                        self.outputs = [tf.nn.xw_plus_b(output,
                                                        output_projection[0],
                                                        output_projection[1])
                                        for output in self.outputs]

            else:
                self.outputs, self.states = seq2seq_f(encoder_inputs=self.encoder_inputs,
                                                      decoder_inputs=self.decoder_inputs[:-1],
                                                      do_decode=False)

                self.losses = seq2seq.sequence_loss(logits=self.outputs,
                                                    targets=targets,
                                                    weights=self.target_weights[:-1],
                                                    softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the models.
        params = tf.trainable_variables()
        self.params = params  # Hold onto this for Woz

        if not forward_only:

            self.updates = []
            self.gradient_norms = []
            opt = tf.train.AdagradOptimizer(self.learning_rate)
            # opt = tf.train.AdamOptimizer(self.learning_rate)

            if self.buckets:
                for b in xrange(len(self.buckets)):
                    gradients = tf.gradients(self.losses[b], params, aggregation_method=2)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                     max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params),
                                                            global_step=self.global_step))
            else:
                gradients = tf.gradients(self.losses, params, aggregation_method=2)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms = norm
                self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only, beam_search):
        """Run a step of the models feeding the given inputs.

        Args:
          session: Tensorflow session to use.
          encoder_inputs: List of numpy int vectors to feed as encoder inputs.
          decoder_inputs: List of numpy int vectors to feed as decoder inputs.
          target_weights: List of numpy float vectors to feed as target weights.
          forward_only: Whether to do the backward step or only forward.
          bucket_id: Which bucket of the models to use, if the models is bucketed.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of enconder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Check if the sizes match.
        try:
            encoder_size, decoder_size = self.buckets[bucket_id]

            if len(encoder_inputs) != encoder_size:
                raise ValueError("Encoder length must be equal to the one in bucket,"
                                 " %d != %d." % (len(encoder_inputs), encoder_size))
            if len(decoder_inputs) != decoder_size:
                raise ValueError("Decoder length must be equal to the one in bucket,"
                                 " %d != %d." % (len(decoder_inputs), decoder_size))
            if len(target_weights) != decoder_size:
                raise ValueError("Weights length must be equal to the one in bucket,"
                                 " %d != %d." % (len(target_weights), decoder_size))
        except (AttributeError, TypeError):
            encoder_size, decoder_size = self.max_sentence_length, self.max_sentence_length

            if len(encoder_inputs) != encoder_size:
                raise ValueError("Encoder length must be equal to max sentence length")
            if len(decoder_inputs) != decoder_size:
                raise ValueError("Decoder length must be equal to max sentence length")
            if len(target_weights) != decoder_size:
                raise ValueError("Target weights must be equal to max sentence length")

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if self.buckets:
            if not forward_only:
                output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                               self.gradient_norms[bucket_id],  # Gradient norm.
                               self.losses[bucket_id]]  # Loss for this batch.
            else:
                if beam_search:
                    # TODO: Make this work
                    print(bucket_id)
                    output_feed = [self.outputs[bucket_id], self.states[bucket_id]]
                else:
                    output_feed = [self.losses[bucket_id]]  # Loss for this batch.

                    for l in xrange(decoder_size):  # Output logits.
                        output_feed.append(self.outputs[bucket_id][l])

        else:
            if not forward_only:
                output_feed = [self.updates,  # Update Op that does SGD.
                               self.gradient_norms,  # Gradient norm.
                               self.losses]  # Loss for this batch.
            else:
                if beam_search:
                    output_feed = [self.outputs, self.states]
                else:
                    output_feed = [self.losses]  # Loss for this batch.

                    for l in xrange(decoder_size):  # Output logits.
                        output_feed.append(self.outputs[l])

        # NOTE: added final arg for summaries
        # output_feed.append(merged_summaries)
        outputs = session.run(output_feed, input_feed)

        # NOTE: added outputs[-1] prefix with summaries to both exit paths
        if not forward_only:
            return outputs[-1], outputs[1], outputs[2], None  # Summaries, Gradient norm, loss, no outputs.
        else:
            return None, None, outputs[0], outputs[1:]  # Summaries, No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """Get a random batch of preprocess from the specified bucket, prepare for step.

        To feed preprocess in step(..) it must be a list of batch-major vectors, while
        preprocess here contains single length-major cases. So the main logic of this
        function is to re-index preprocess cases to be in the proper format for feeding.

        Args:
          If using buckets:
            data: A tuple of size len(self.buckets) in which each element contains
              lists of pairs of input and output preprocess that we use to create a batch.
            bucket_id: Integer, which bucket to get the batch for.
          Else:
            preprocess: The entire training set.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        try:
            encoder_size, decoder_size = self.buckets[bucket_id]
        except (TypeError, AttributeError):
            encoder_size = self.max_sentence_length
            decoder_size = self.max_sentence_length

        encoder_inputs = list()
        decoder_inputs = list()

        # Get a random batch of encoder and decoder inputs from preprocess,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            try:
                encoder_input, decoder_input = random.choice(data[bucket_id])
            except (TypeError, IndexError):
                encoder_input, decoder_input = random.choice(data)

            # Encoder inputs are padded and then reversed.
            encoder_pad = [basic.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([basic.GO_ID] + decoder_input + [basic.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the preprocess selected above.
        batch_weights = list()
        batch_encoder_inputs = list()
        batch_decoder_inputs = list()

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == basic.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
