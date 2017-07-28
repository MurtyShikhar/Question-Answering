from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from general_utils import Progbar
from data_utils import *
from collections import defaultdict as ddict

from attention_wrapper import _maybe_mask_score
from attention_wrapper import *
from evaluate import exact_match_score, f1_score
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import array_ops


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(stream = sys.stdout, level=logging.INFO)

# -- A helper function to reverse a tensor along seq_dim
def _reverse(input_, seq_lengths, seq_dim, batch_dim):
  if seq_lengths is not None:
    return array_ops.reverse_sequence(
        input=input_, seq_lengths=seq_lengths,
        seq_dim=seq_dim, batch_dim=batch_dim)
  else:
    return array_ops.reverse(input_, axis=[seq_dim])



class Encoder(object):
    def __init__(self, hidden_size, initializer = lambda : None):#tf.contrib.layers.xavier_initializer):
        self.hidden_size = hidden_size
        self.init_weights = initializer


    def encode(self, inputs, masks, encoder_state_input = None):
        """
        :param inputs: vector representations of question and passage (a tuple) 
        :param masks: masking sequences for both question and passage (a tuple)

        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of the question and passage.
        """


        question, passage = inputs
        masks_question, masks_passage = masks    


        # read passage conditioned upon the question
        with tf.variable_scope("encoded_question"):
            lstm_cell_question = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            encoded_question, (q_rep, _) = tf.nn.dynamic_rnn(lstm_cell_question, question, masks_question, dtype=tf.float32) # (-1, Q, H)

        with tf.variable_scope("encoded_passage"):
            lstm_cell_passage  = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            encoded_passage, (p_rep, _) =  tf.nn.dynamic_rnn(lstm_cell_passage, passage, masks_passage, dtype=tf.float32) # (-1, P, H)

        return encoded_question, encoded_passage , q_rep, p_rep


class BaselineDecoder(object):
    def __init__(self):
        return

    def decode(self, encoded_passage , q_rep, mask, labels):

        # (batch_size, 1, D),  (batch_size, Q, D)
        input_size = q_rep.get_shape()[-1]
        q_rep1 = tf.layers.dense(q_rep, input_size, name="W1")
        q_rep2 = tf.layers.dense(q_rep, input_size, name="W2")


        q_rep1 = tf.expand_dims(q_rep1, 1)
        logit_1 = tf.reduce_sum(q_rep1*encoded_passage, [2])

        q_rep2 = tf.expand_dims(q_rep2, 1)
        logit_2 = tf.reduce_sum(q_rep2*encoded_passage, [2])

        func = lambda score: _maybe_mask_score(score, mask, float("-inf"))

        return [func(logit_1),func(logit_2)]


   
class Decoder(object):
    def __init__(self, hidden_size, initializer= lambda : None):
        self.hidden_size = hidden_size
        self.init_weights = initializer


    def run_match_lstm(self, encoded_rep, masks):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks

        match_lstm_cell_attention_fn = lambda curr_input, state : tf.concat([curr_input, state], axis = -1)
        query_depth = encoded_question.get_shape()[-1]


        # output attention is false because we want to output the cell output and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            #scope = tf.VariableScope(name = "rnn",  reuse=True)

            attention_mechanism_match_lstm = BahdanauAttention(query_depth, encoded_question, memory_sequence_length = masks_question)
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            lstm_attender  = AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention = False, attention_input_fn = match_lstm_cell_attention_fn)

            # we don't mask the passage because masking the memories will be handled by the pointerNet
            reverse_encoded_passage = _reverse(encoded_passage, masks_passage, 1, 0)

            output_attender_fw, _ = tf.nn.dynamic_rnn(lstm_attender, encoded_passage, dtype=tf.float32, scope ="rnn")    
            output_attender_bw, _ = tf.nn.dynamic_rnn(lstm_attender, reverse_encoded_passage, dtype=tf.float32, scope = "rnn")

            output_attender_bw = _reverse(output_attender_bw, masks_passage, 1, 0)

        
        output_attender = tf.concat([output_attender_fw, output_attender_bw], axis = -1) # (-1, P, 2*H)
        return output_attender


    def run_answer_ptr(self, output_attender, masks, labels):
        batch_size = tf.shape(output_attender)[0]
        masks_question, masks_passage = masks
        labels = tf.unstack(labels, axis=1) 
        #labels = tf.ones([batch_size, 2, 1])


        answer_ptr_cell_input_fn = lambda curr_input, context : context # independent of question
        query_depth_answer_ptr = output_attender.get_shape()[-1]

        with tf.variable_scope("answer_ptr_attender"):
            attention_mechanism_answer_ptr = BahdanauAttention(query_depth_answer_ptr , output_attender, memory_sequence_length = masks_passage)
            # output attention is true because we want to output the attention values
            cell_answer_ptr = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True )
            answer_ptr_attender = AttentionWrapper(cell_answer_ptr, attention_mechanism_answer_ptr, cell_input_fn = answer_ptr_cell_input_fn)
            logits, _ = tf.nn.static_rnn(answer_ptr_attender, labels, dtype = tf.float32)

        return logits 



    def decode(self, encoded_rep, q_rep, masks, labels):
        """
        takes in encoded_rep
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param encoded_rep: 
        :param masks
        :param labels


        :return: logits: for each word in passage the probability that it is the start word and end word.
        """

        output_attender = self.run_match_lstm(encoded_rep, masks)
        logits = self.run_answer_ptr(output_attender, masks, labels)
    
        return logits
    




class QASystem(object):    
    def __init__(self, encoder, decoder, pretrained_embeddings, config):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up logging ======


        logger = logging.getLogger("QASystemLogger")
        self.logger = logger


        # ==== set up placeholder tokens ========
        self.embeddings = pretrained_embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        self.setup_placeholders()



        # ==== assemble pieces ====
        with tf.variable_scope("qa"):
            self.setup_word_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_train_op()
            self.saver = tf.train.Saver()

        



    def setup_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            adam_optimizer = tf.train.AdamOptimizer()
            grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))

            clip_val = self.config.max_gradient_norm
            # if -1 then do not perform gradient clipping
            if clip_val != -1:
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
                self.global_grad = tf.global_norm(clipped_grads)
                self.gradients = zip(clipped_grads, vars)
            else:
                self.global_grad = tf.global_norm(grads)
                self.gradients = zip(grads, vars)


            self.train_op = adam_optimizer.apply_gradients(self.gradients)

        self.init = tf.global_variables_initializer()


    def get_feed_dict(self, questions, contexts, answers, dropout_val):
        """
        -arg questions: A list of list of ids representing the question sentence
        -arg contexts: A list of list of ids representing the context paragraph
        -arg dropout_val: A float representing the keep probability for dropout 

        :return: dict {placeholders: value}
        """

        padded_questions, question_lengths = pad_sequences(questions, 0)
        padded_contexts, passage_lengths = pad_sequences(contexts, 0)


        feed = {
            self.question_ids : padded_questions,
            self.passage_ids : padded_contexts,
            self.question_lengths : question_lengths,
            self.passage_lengths : passage_lengths,
            self.labels : answers,
            self.dropout : dropout_val
        }

        return feed


    def setup_word_embeddings(self):
        '''
            Create an embedding matrix (initialised with pretrained glove vectors and updated only if self.config.train_embeddings is true)
            lookup into this matrix and apply dropout (which is 1 at test time and self.config.dropout at train time)
        '''
        with tf.variable_scope("vocab_embeddings"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable= self.config.train_embeddings)
            question_emb = tf.nn.embedding_lookup(_word_embeddings, self.question_ids, name = "question") # (-1, Q, D)
            passage_emb = tf.nn.embedding_lookup(_word_embeddings, self.passage_ids, name = "passage") # (-1, P, D)
            # Apply dropout
            self.question = tf.nn.dropout(question_emb, self.dropout)
            self.passage  = tf.nn.dropout(passage_emb, self.dropout)
            



    def setup_placeholders(self):
        self.question_ids = tf.placeholder(tf.int32, shape = [None, None], name = "question_ids")
        self.passage_ids = tf.placeholder(tf.int32, shape = [None, None], name = "passage_ids")

        self.question_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
        self.passage_lengths = tf.placeholder(tf.int32, shape = [None], name = "passage_lengths")

        self.labels = tf.placeholder(tf.int32, shape = [None, 2], name = "gold_labels")
        self.dropout = tf.placeholder(tf.float32, shape=[], name = "dropout")

    def setup_system(self):
        """
           Apply the encoder to the question and passage embeddings. Follow that up by Match-LSTM and Answer-Ptr 
        """
        encoder = self.encoder
        decoder = self.decoder
        encoded_question, encoded_passage, q_rep, p_rep = encoder.encode([self.question, self.passage], [self.question_lengths, self.passage_lengths],
                                                             encoder_state_input = None)

        logits= decoder.decode([encoded_question, encoded_passage], q_rep, [self.question_lengths, self.passage_lengths], self.labels)

        self.logits = logits


    def setup_loss(self):
        """
        self.logits are the 2 sets of logit (num_classes) values for each example, masked with float(-inf) beyond the true sequence length
        :return: Loss for the current batch of examples
        """
      
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[0], labels=self.labels[:,0])
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[1], labels=self.labels[:,1])
        self.loss = tf.reduce_mean(losses)


    def initialize_model(self, session, train_dir):
        """
            param: session managed from train.py
            param: train_dir : the directory in which models are saved

        """
        ckpt = tf.train.get_checkpoint_state(train_dir, latest_filename ="best_model.chk")
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            self.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            self.logger.info("Created model with fresh parameters.")
            session.run(self.init)                
            self.logger.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))




    def test(self, session, valid):
        """
        valid: a list containing q, c and a.
        :return: loss on the valid dataset and the logit values
        """
        
        q, c, a = valid

        # at test time we do not perform dropout.
        input_feed =  self.get_feed_dict(q, c, a, 1.0)

        output_feed = [self.logits, self.loss]

        outputs, loss = session.run(output_feed, input_feed)

        return outputs[0], outputs[1], loss


    def answer(self, session, dataset):
        '''
            Get the answers for dataset
        '''

        q, c, a = zip(*[[_q, _c, _a] for (_q, _c, _a) in dataset])

        yp, yp2, loss = self.test(session, [q, c, a])


    
        # -- Boundary Model with a max span restriction of 15
        
        def func(y1, y2):
            max_ans = -999999
            a_s, a_e= 0,0
            num_classes = len(y1)
            for i in xrange(num_classes):
                for j in xrange(15):
                    if i+j >= num_classes:
                        break

                    curr_a_s = y1[i];
                    curr_a_e = y2[i+j]
                    if (curr_a_e+curr_a_s) > max_ans:
                        max_ans = curr_a_e + curr_a_s
                        a_s = i
                        a_e = j

            return (a_s, a_e)


        a_s, a_e = [], []
        for i in xrange(yp.shape[0]):
            _a_s, _a_e = func(yp[i], yp2[i])
            a_s.append(_a_s)
            a_e.append(_a_e)
 

        return (np.array(a_s), np.array(a_e))


    def evaluate_model(self, session, dataset):
        """

    
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :return: exact match scores
        """

        sample = len(dataset)
        a_s, a_o = self.answer(session, dataset)
        answers = np.hstack([a_s.reshape([sample, -1]), a_o.reshape([sample,-1])])
        gold_answers = np.array([a for (_,_, a) in dataset])

        em = np.sum(answers == gold_answers)/float(len(answers))


        return em


    def run_epoch(self, session, train):
        """
        Perform one complete pass over the training data and evaluate on dev
        """
     
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)


        for i, (q_batch, c_batch, a_batch) in enumerate(minibatches(train, self.config.batch_size)):

            # at training time, dropout needs to be on.
            input_feed = self.get_feed_dict(q_batch, c_batch, a_batch, self.config.dropout_val)

            _, train_loss = session.run([self.train_op, self.loss], feed_dict=input_feed)
            prog.update(i + 1, [("train loss", train_loss)])




    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        :param session: it should be passed in from train.py
        :param dataset: a list containing the training and dev data
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        if not tf.gfile.Exists(train_dir):
            tf.gfile.MkDir(train_dir)


        train, dev = dataset

        # ====== Load a pretrained model if it exists or create a new one if no pretrained available ======
        self.initialize_model(session, train_dir)

        em = self.evaluate_model(session, dev)
        self.logger.info("#-----------Initial Exact match on dev set: %5.4f ---------------#" %em)
        #self.logger.info("#-----------Initial F1 on dev set: %5.4f ---------------#" %f1)

        best_em = 0

        for epoch in xrange(self.config.num_epochs):
            self.logger.info("*********************EPOCH: %d*********************" %(epoch+1))
            self.run_epoch(session, train)
            em = self.evaluate_model(session, dev)
            self.logger.info("#-----------Exact match on dev set: %5.4f #-----------" %em)
            #self.logger.info("#-----------F1 on dev set: %5.4f #-----------" %f1)

            #======== Save model if it is the best so far ========
            if (em > best_em):
                self.saver.save(session, "%s/best_model.chk" %train_dir)
                best_em = em

