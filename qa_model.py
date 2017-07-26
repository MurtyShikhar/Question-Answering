from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from general_utils import Progbar
from data_utils import *
from collections import defaultdict as ddict

from attention_wrapper import _maybe_mask_score
from attention_wrapper import *
#from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from evaluate import exact_match_score, f1_score
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import array_ops


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)



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
            encoded_question, (last_state, _) = tf.nn.dynamic_rnn(lstm_cell_question, question, masks_question, dtype=tf.float32) # (-1, Q, H)

        with tf.variable_scope("encoded_passage"):
            lstm_cell_passage  = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            encoded_passage, _ =  tf.nn.dynamic_rnn(lstm_cell_passage, passage, masks_passage, dtype=tf.float32) # (-1, P, H)

        return encoded_question, last_state , encoded_passage

   
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
        labels = [tf.ones([batch_size, 1]), tf.ones([batch_size,1]) ]
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

        # ==== set up training/updating procedure ====
        

    def setup_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            adam_optimizer = tf.train.AdamOptimizer()
            grads, vars = zip(*adam_optimizer.compute_gradients(self.loss))
            #clipped_grads, _ = tf.clip_by_global_norm(grads, 4.0)

            self.global_grad = tf.global_norm(grads)
            self.gradients = zip(grads, vars)

            self.train_op = adam_optimizer.apply_gradients(self.gradients)

        self.init = tf.global_variables_initializer()


    def get_feed_dict(self, questions, contexts, answers):
        """
        :return: dict {placeholders: value}
        """

        padded_questions, question_lengths = pad_sequences(questions, 0)
        padded_contexts, passage_lengths = pad_sequences(contexts, 0)


        feed = {
            self.question_ids : padded_questions,
            self.passage_ids : padded_contexts,
            self.question_lengths : question_lengths,
            self.passage_lengths : passage_lengths,
            self.labels : answers
        }

        return feed


    def setup_word_embeddings(self):
        with tf.variable_scope("vocab_embeddings"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable= self.config.train_embeddings)
            self.question = tf.nn.embedding_lookup(_word_embeddings, self.question_ids, name = "question") # (-1, Q, D)
            self.passage = tf.nn.embedding_lookup(_word_embeddings, self.passage_ids, name = "passage") # (-1, P, D)



    def setup_placeholders(self):
        self.question_ids = tf.placeholder(tf.int32, shape = [None, None], name = "question_ids")
        self.passage_ids = tf.placeholder(tf.int32, shape = [None, None], name = "passage_ids")

        self.question_lengths = tf.placeholder(tf.int32, shape=[None], name="question_lengths")
        self.passage_lengths = tf.placeholder(tf.int32, shape = [None], name = "passage_lengths")

        self.labels = tf.placeholder(tf.int32, shape = [None, None], name = "gold_labels")


    def setup_system(self):
        """
            DOCUMENTATION TODO
        """
        encoder = self.encoder
        decoder = self.decoder
        encoded_question, q_rep, encoded_passage = encoder.encode([self.question, self.passage], [self.question_lengths, self.passage_lengths],
                                                             encoder_state_input = None)

        logits= decoder.decode([encoded_question, encoded_passage], q_rep, [self.question_lengths, self.passage_lengths], self.labels)

        self.logits = logits


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """

        
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[0], labels=self.labels[:,0])
        losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[1], labels=self.labels[:,1])
        self.loss = tf.reduce_mean(losses)


    def test(self, session, valid):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        
        dat = [dat for dat in minibatches(valid, len(valid))]
        q, c, a = dat[0]
        input_feed =  self.get_feed_dict(q, c, a)

        output_feed = [self.logits, self.loss]

        outputs, loss = session.run(output_feed, input_feed)
        print(outputs.shape)

        return outputs[:, 0], outputs[:, 1], loss


    def answer(self, session, dataset):

        yp, yp2, loss = self.test(session, dataset)

        max_ans = -999999
        a_s, a_e= 0,0
        for i in xrange(yp.shape[1]):
            for j in xrange(15):
                if i+j > yp.shape[1]:
                    break

                curr_a_s = yp[i];
                curr_a_e = yp2[i+j]
                if (curr_a_e+curr_a_s) > max_ans:
                    max_ans = curr_a_e + curr_a_s
                    a_s = curr_a_s
                    a_e = curr_a_e

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """

        _, _, valid_cost = self.test(sess, valid_dataset)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=10, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1=0
        dat = []; i =0 
        for data in dataset:
            if (i >= sample): 
                break
            else:
                dat.append(data)
                i += 1

        a_s, a_o = self.answer(session, dat)

        answers = np.hstack([a_s.reshape([sample, -1]), a_o.reshape([sample,-1])])
        gold_answers = np.array([a for (_,_, a) in dat])
        print(answers)
        print(gold_answers) 
        em = np.sum(answers == gold_answers)/float(len(answers))

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def run_epoch(self, session, train, dev, epoch):
        """
        Perform one complete pass over the training data and evaluate on dev
        """
        #f1, em = self.evaluate_answer(session, dev)
        #print("Exact match on dev set:",em)

        writer = tf.summary.FileWriter("qa_model1")
        writer.add_graph(session.graph)
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        #prog = Progbar(target=nbatches)

        grad_graph = ddict(list)
        train_loss_graph = []
        global_grad_graph = []

        for i, (q_batch, c_batch, a_batch) in enumerate(minibatches(train, self.config.batch_size)):
            input_feed = self.get_feed_dict(q_batch, c_batch, a_batch)

            #gradients = tf.gradients(self.loss, tf.trainable_variables())

            _, gradients, train_loss, global_grad = session.run([self.train_op, self.gradients, self.loss, self.global_grad], feed_dict=input_feed)
            print("=======LOSS: ",train_loss, "=============GRAD: ", global_grad, "======")
        
            global_grad_graph.append(global_grad)
            train_loss_graph.append(train_loss)

            for grad, sym_grad in zip(gradients, self.gradients):
                var_name = ":".join(sym_grad[1].name.split("/"))
                grad_graph[var_name].append(np.linalg.norm(grad[0]))
                f = open("params/%s" %var_name, "w")
                np.savetxt(f,grad[0])
                f.close()
                #print(np.linalg.norm(grad[0]), sym_grad[1].name)
        

        return grad_graph, train_loss_graph, global_grad_graph        

    def construct_graph(self, grad_graph, train_loss_graph, global_grad_graph, epoch):

        num_figs = len(grad_graph) + 2
        plt.figure(num_figs*epoch)
        plt.plot(train_loss_graph)
        plt.savefig("figs/%s-train_loss_graph.png" %epoch)
        plt.figure(num_figs*epoch+1)
        plt.plot(global_grad_graph)
        plt.savefig("figs/%s-global_grad_graph.png" %epoch)

        for i,key in enumerate(grad_graph):
            plt.figure(num_figs*epoch + 2+i)
            plt.plot(grad_graph[key])
            plt.savefig("figs/%s-%s.png" %(epoch,key))



    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        train, dev = dataset
        #session = tf_debug.LocalCLIDebugWrapperSession(session)
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        print("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        train_loss = []
        for epoch in xrange(self.config.num_epochs):
            print("*********************EPOCH: %d*********************" %(epoch+1))
            session.run(self.init)
            grad_graph, train_loss_graph, global_grad_graph = self.run_epoch(session, train, dev, epoch)

            train_loss += train_loss_graph
            self.construct_graph( grad_graph, train_loss, global_grad_graph, epoch)
