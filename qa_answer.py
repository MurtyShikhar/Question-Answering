from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin
from config import Config


from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data
from data_utils import *

import logging
logging.basicConfig(level=logging.INFO)





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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data



def prepare_dev2(config):
    dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)



def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)


    def normalize(dat):
        return map(lambda tok: map(int, tok.split()), dat)

    context_data = normalize(context_data)
    question_data = normalize(question_data)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, dataset, uuid_data, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    q,c,a = dataset
    num_points = len(a)
    sample_size = 1000


    answers_canonical = []
    num_iters = int((num_points+sample_size-1)/sample_size)

    for i in xrange(num_iters):
        curr_slice_st = i*sample_size
        curr_slice_en = min((i+1)*sample_size, num_points)

        slice_sz = curr_slice_en - curr_slice_st 

        q_curr = q[curr_slice_st : curr_slice_en]
        c_curr = c[curr_slice_st : curr_slice_en]
        a_curr = a[curr_slice_st : curr_slice_en]

        s, e = model.answer(sess, [q_curr, c_curr, a_curr])

        for j in xrange(slice_sz):
            st_idx = s[j]
            en_idx = e[j]
            curr_context = c[curr_slice_st+j]
            curr_uuid = uuid_data[curr_slice_st+j]

            curr_ans = ""
            for idx in xrange(st_idx, en_idx+1):
                curr_tok = curr_context[idx]
                curr_ans += " %s" %(rev_vocab[curr_tok])


            answers[curr_uuid] = curr_ans
            answers_canonical.append((s,e))


    return answers, answers_canonical



def run_func2(dataset, config):
    vocab, rev_vocab = initialize_vocab(config.vocab_path)


    q, c, a = zip(*[[_q, _c, _a] for (_q, _c, _a) in dataset])

    dataset = [q, c, a]

    embed_path = config.embed_path

    embeddings = get_trimmed_glove_vectors(embed_path)


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)

    qa = QASystem(encoder, decoder, embeddings, config)
    question_uuid_data = [i for i in xrange(len(a))]
    
    with tf.Session() as sess:
        qa.initialize_model(sess, config.train_dir)
        answers, answers_canonical = generate_answers(sess, qa, dataset, question_uuid_data, rev_vocab)
        # write to json file to root dir
        with io.open('dev-prediction.txt', 'w', encoding='utf-8') as f:
            for i in xrange(len(a)):
                curr_ans = unicode(answers[i], "utf-8")
                f.write("%s\n" %(curr_ans))

        #get_numbers(ans)



def run_func():
    config = Config()

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way
    vocab, rev_vocab = initialize_vocab(config.vocab_path)

    dev_path = "data/squad/train-v1.1.json"
    dev_dirname = os.path.dirname(os.path.abspath(dev_path))
    dev_filename = os.path.basename(dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)


    
    ques_len = len(question_data)
    answers = [[0, 0] for _ in xrange(ques_len)]

    dataset = [question_data, context_data, answers]

    embed_path = config.embed_path

    embeddings = get_trimmed_glove_vectors(embed_path)


    encoder = Encoder(config.hidden_state_size)
    decoder = Decoder(config.hidden_state_size)

    qa = QASystem(encoder, decoder, embeddings, config)
    
    with tf.Session() as sess:
        qa.initialize_model(sess, config.train_dir)
        answers, _ = generate_answers(sess, qa, dataset, question_uuid_data, rev_vocab)
        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))



if __name__ == "__main__":
    #config = Config()
    #dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)

    run_func()
