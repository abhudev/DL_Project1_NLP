
# coding: utf-8

# In this notebook, we will learn concretely how to build a neural language model (LM)
# 
# The code has been adapted from the [official tutorial on using eager for LM](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py)
# 




import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.ops import lookup_ops
from collections import OrderedDict
import argparse
import data_feed

parser = argparse.ArgumentParser()
parser.add_argument('--nli_data', type=str, default='../PROJECT_data/NLI/allnli.train.txt.clean.noblank')
parser.add_argument('--nmt_en', type=str, default='../PROJECT_data/NMT/nmt.de-en.en.tok')
parser.add_argument('--nmt_de', type=str, default='../PROJECT_data/NMT/nmt.de-en.de.tok')
parser.add_argument('--parse_data', type=str)
parser.add_argument('--en_vocab_file',type=str, default='../PROJECT_data/en_vocab.txt')
parser.add_argument('--de_vocab_file',type=str, default='../PROJECT_data/de_vocab.txt')
parser.add_argument('--rnn_cell', type=str, default='gru')
# parser.add_argument('--vocab_size', type=str, default=30000)
args = parser.parse_args()

tf.enable_eager_execution()
tf.set_random_seed(42)

# Parameters of model
embed_size = 256
hidden_size = 512
dropout_prob = 0.3
adam_lr = 0.002
batch_size = 32
vocab_size = 30000 # vocab_table.size()

# Embedding model
class Embedding(tf.keras.Model):
    def __init__(self, V, d):
        super(Embedding, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
    
    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)

class StaticRNN(tf.keras.Model):
    def __init__(self, h, cell):
        super(StaticRNN, self).__init__()
        if cell == 'lstm':
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=h)
        elif cell == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=h)
        elif cell == 'vanilla':
            self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=h)
        else:
            assert(False)
        
        
    def call(self, word_vectors, num_words):
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        outputs, final_state = tf.nn.static_rnn(cell=self.cell, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32)
        return outputs

class SharedEncoder(tf.keras.Model):
    def __init__(self, V, word_dim, hidden_size, cell_type):
        super(SharedEncoder, self).__init__()
        self.word_embedding = Embedding(V, word_dim)
        self.cell_type = cell_type
        if(self.cell_type == 'gru'):
            self.encoder = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        elif(self.cell_type = 'lstm'):
            self.encoder = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        elif(self.cell_type == 'vanilla'):
            self.encoder = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
        else:
            assert(False)
        
    def call(self, word_vectors, num_words):
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        _, final_state = tf.nn.static_rnn(cell=self.encoder_cell, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32)
        return final_state

class NMTDecoder(tf.keras.Model):
    def __init__(self, V, word_dim, hidden_size, cell_type):
        super(NMTDecoder, self).__init__()
        self.word_embedding = Embedding(V, word_dim)
        self.cell_type = cell_type
        if(self.cell_type == 'gru'):
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        elif(self.cell_type = 'lstm'):
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        elif(self.cell_type == 'vanilla'):
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
        else:
            assert(False)
        self.decoder = tf.contrib.seq2seq.BeamSearchDecoder(self.rnn_cell, self.word_embedding, <start_tokens>, '<eos>', <initial_state>, <beam_width>, <output_layer>, )
    
    def call(word_vectors, num_words):
        pass
        # Can initialize

# An MLP for NLI
class Nli_MLP(tf.keras.Model):
    def __init__(self, drop_p, hidden_dim, num_classes=3):
        self.drop_p = drop_p
        self.layer_size = hidden_dim*4 # Concatenate four vectors
        self.num_classes = num_classes
        # TODO - add dropout
        # TODO - add seed?
        self.dropout = tf.keras.layers.Dropout(drop_p)
        self.mlp_in = tf.keras.layers.Dense(units=self.layer_size, activation='relu', kernel_initializer='random_normal', bias_initializer='uniform')
        self.mlp_out = tf.keras.layers.Dense(units=self.num_classes, kernel_initializer='random_normal')
        self.classes_out = tf.keras.Softmax()
    def call(self, input):
        l1_drop = self.dropout(input)
        l1 = self.mlp_in(l1_drop)
        l2 = self.mlp_out(l1)
        output = self.classes_out(l2)
        return classes_out


class MultiModel(tf.keras.Model):
    def __init__(self, V, word_dim, hidden_size):
        super(MultiModel, self).))__init__()
        self.encoder = SharedEncoder(V, word_dim, hidden_size)
        self.mtDecoder = NMTDecoder(V, word_dim, hidden_size)
        self.parseDecoder = ParseDecoder(V, word_dim, hidden_size)
    
    # Take care - which task?
    # Parameters:
    # datum - sentence input
    # num_words - input of number of words in each sentence
    # datunm2 - second optional sentence for NLI
    # num_word2 - NLI
    def call(datum, num_words, task):
        if(task == 'nmt'):
            pass
        elif task == 'parse':
            pass
        elif task == 'nli':
            pass
        else:
            assert(False)
        
        

class LanguageModel(tf.keras.Model):
    def __init__(self, V, d, h, cell):
        super(LanguageModel, self).__init__()
        self.word_embedding = Embedding(V, d)
        self.rnn = StaticRNN(h, cell)
        self.output_layer = tf.keras.layers.Dense(units=V)
        
    def call(self, datum):
        word_vectors = self.word_embedding(datum[0])
        rnn_outputs_time = self.rnn(word_vectors, datum[2])
        
        #We want to convert it back to shape batch_size x TimeSteps x h
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        logits = self.output_layer(rnn_outputs)
        return logits


en_vocab_table = lookup_ops.index_table_from_file(args.en_vocab_file)
de_vocab_table = lookup_ops.index_table_from_file(args.de_vocab_file)

