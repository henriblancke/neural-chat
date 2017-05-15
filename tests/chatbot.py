import os
import numpy as np
import tensorflow as tf
from models import helper
from variables import params
from utils.preprocess import basic


class ChatBot(object):
    """
    ChatBot for interactive conversations
    """

    def __init__(self):
        self.beam = params.beam_search
        self.beam_size = params.beam_size
        self.sess = tf.Session()
        self.model = helper.init_model(self.sess,
                                       forward_only=True)
        self.model.batch_size = 1  # Respond 1 sentence at a time.

        vocab_path = os.path.join(params.data_dir, "vocab%d" % params.vocab_size)
        self.vocab, self.rev_vocab = basic.initialize_vocabulary(vocab_path)

    def vocab_lookup(self, out):
        if out < len(self.rev_vocab):
            return self.rev_vocab[out]
        else:
            return basic._UNK

    def greedy_decoder(self, encoder_inputs, decoder_inputs, target_weights, bucket_id):
        # Get output logits for the sentence.
        _, _, _, output_logits = self.model.step(session=self.sess,
                                                 encoder_inputs=encoder_inputs,
                                                 decoder_inputs=decoder_inputs,
                                                 target_weights=target_weights,
                                                 bucket_id=bucket_id,
                                                 forward_only=True,
                                                 beam_search=False)

        output = [int(np.argmax(logit, axis=1)) for logit in output_logits]

        # If there is an EOS symbol in outputs, cut them at that point.
        if basic.EOS_ID in output:
            output = output[:output.index(basic.EOS_ID)]

        sentence = " ".join([self.vocab_lookup(w) for w in output])

        return [sentence]

    # TODO: Implement Anti-LM
    def beam_search(self, encoder_inputs, decoder_inputs, target_weights, bucket_id):

        _, _, beams, probs = self.model.step(session=self.sess,
                                             encoder_inputs=encoder_inputs,
                                             decoder_inputs=decoder_inputs,
                                             target_weights=target_weights,
                                             bucket_id=bucket_id,
                                             forward_only=True,
                                             beam_search=True)

        results = list()
        for beam, prob in zip(beams, probs[0]):
            beam = beam.tolist()
            results.append((" ".join([self.vocab_lookup(w) for w in beam[:beam.index(basic.EOS_ID)]]), prob))

        return results

    def respond(self, sentence):
        # Get token-ids for the input sentence.
        token_ids = basic.sentence_to_token_ids(sentence, self.vocab)

        try:
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(self.model.buckets))
                             if self.model.buckets[b][0] > len(token_ids)])

            outputs = []
            feed_data = {bucket_id: [(token_ids, outputs)]}

            # Get a 1-element batch to feed the sentence to the models.
            encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(data=feed_data,
                                                                                  bucket_id=bucket_id)

            if self.beam:
                return self.beam_search(encoder_inputs=encoder_inputs,
                                        decoder_inputs=decoder_inputs,
                                        target_weights=target_weights,
                                        bucket_id=bucket_id)
            else:
                return self.greedy_decoder(encoder_inputs=encoder_inputs,
                                           decoder_inputs=decoder_inputs,
                                           target_weights=target_weights,
                                           bucket_id=bucket_id)

        except (AttributeError, TypeError):
            encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(data=[(token_ids, [])],
                                                                                  bucket_id=None)

            if self.beam:
                return self.beam_search(encoder_inputs=encoder_inputs,
                                        decoder_inputs=decoder_inputs,
                                        target_weights=target_weights,
                                        bucket_id=None)
            else:
                return self.greedy_decoder(encoder_inputs=encoder_inputs,
                                           decoder_inputs=decoder_inputs,
                                           target_weights=target_weights,
                                           bucket_id=None)
