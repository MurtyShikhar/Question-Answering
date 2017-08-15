import os
import json

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder, BaselineDecoder
from config import Config
from data_utils import *
from os.path import join as pjoin


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



def run_func():
    config = Config()
    train = squad_dataset(config.question_train, config.context_train, config.answer_train)
    dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)


    embed_path = config.embed_path
    vocab_path = config.vocab_path
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embeddings = get_trimmed_glove_vectors(embed_path)


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)

    qa = QASystem(encoder, decoder, embeddings, config)
    
    with tf.Session() as sess:
        # ====== Load a pretrained model if it exists or create a new one if no pretrained available ======
        qa.initialize_model(sess, config.train_dir)
        qa.train(sess, [train, dev], config.train_dir)



if __name__ == "__main__":
    run_func()
