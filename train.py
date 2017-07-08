from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from config import Config
from data_utils import *
from os.path import join as pjoin


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def run_func():
    config = Config()
    train = squad_dataset(config.question_train, config.context_train, config.answer_train)
    dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)

    dev_small = [data for data in dev][:5000]

    embed_path = config.embed_path
    vocab_path = config.vocab_path
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embeddings = get_trimmed_glove_vectors(embed_path)


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)

    qa = QASystem(encoder, decoder, embeddings, config)

    with tf.Session() as sess:
        qa.train(sess, [dev_small, dev_small], "")


# def main(_):

#     # Do what you need to load datasets from FLAGS.data_dir
#     dataset = None

#     embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
#     vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
#     vocab, rev_vocab = initialize_vocab(vocab_path)

#     encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
#     decoder = Decoder(output_size=FLAGS.output_size)

#     qa = QASystem(encoder, decoder)

#     if not os.path.exists(FLAGS.log_dir):
#         os.makedirs(FLAGS.log_dir)
#     file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
#     logging.getLogger().addHandler(file_handler)

#     print(vars(FLAGS))
#     with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
#         json.dump(FLAGS.__flags, fout)

#     with tf.Session() as sess:
#         load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
#         initialize_model(sess, qa, load_train_dir)

#         save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
#         qa.train(sess, dataset, save_train_dir)

#         qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    run_func()
