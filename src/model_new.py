
# coding: utf-8

# In this notebook, we will learn concretely how to build a neural language model (LM)
# 
# The code has been adapted from the [official tutorial on using eager for LM](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py)
# 

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tensorflow.python.ops import lookup_ops
from collections import OrderedDict
import argparse
import data_feed
from tensorflow.python.layers.core import Dense

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--nli_data', type=str, default='../PROJECT_data/NLI/allnli.train.txt.clean.noblank')
parser.add_argument('--nli_premise', type=str, default='../PROJECT_data/NLI/allnli_premise.txt')
parser.add_argument('--nli_hypothesis', type=str, default='../PROJECT_data/NLI/allnli_hypothesis.txt')
parser.add_argument('--nli_class', type=str, default = '../PROJECT_data/NLI/allnli_class.txt')
parser.add_argument('--nmt_en', type=str, default='../PROJECT_data/NMT/nmt.de-en.en.tok')
parser.add_argument('--nmt_de', type=str, default='../PROJECT_data/NMT/nmt.de-en.de.tok')
parser.add_argument('--parse_data', type=str)
parser.add_argument('--en_vocab_file',type=str, default='../PROJECT_data/en_vocab.txt')
parser.add_argument('--de_vocab_file',type=str, default='../PROJECT_data/de_vocab.txt')
parser.add_argument('--nli_class_vocab', type=str, default='../PROJECT_data/NLI/nli_class_vocab.txt')
parser.add_argument('--rnn_cell', type=str, default='gru')
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--bsize', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=300)
# parser.add_argument('--')
# parser.add_argument('--vocab_size', type=str, default=30000)
args = parser.parse_args()


tf.set_random_seed(42)

#--------------------BEGIN DEFINITIONS--------------------#

# Embedding model
class Embedding(tf.keras.Model):
    def __init__(self, V, d):
        super(Embedding, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
    
    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)

# Different kinds of RNN cells to be used
dict_fast_rnn_cells = {'GPU': 
                        {'gru': tf.contrib.cudnn_rnn.CudnnGRU, 
                         'lstm': tf.contrib.cudnn_rnn.CudnnLSTM, 
                         'vanilla':  tf.contrib.cudnn_rnn.CudnnRNNTanh}, 

                  'CPU': 
                        {'gru': tf.contrib.rnn.GRUBlockCellV2, 
                         'lstm': tf.contrib.rnn.LSTMBlockCell, 
                         'vanilla': tf.nn.rnn_cell.BasicRNNCell}
                 }

dict_rnn_cells = {
                    'gru': tf.nn.rnn_cell.GRUCell,
                    'lstm': tf.nn.rnn_cell.LSTMCell,
                    'vanilla': tf.nn.rnn_cell.BasicRNNCell
                 }

# Shared encoder class
class SharedEncoder(tf.keras.Model):
    def __init__(self, V, word_dim, hidden_size, cell_type):
        super(SharedEncoder, self).__init__()
        self.word_embedding = Embedding(V, word_dim)
        self.cell_type = cell_type        
        try:
            self.encoder = dict_rnn_cells[self.cell_type](num_units=hidden_size)
        except:
            assert(False)
        
    # Input - datum[0], datum[1] or datum[2], datum[3]
    def call(self, word_indices, num_words):
        word_vectors = self.word_embedding(word_indices)
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        _, final_state = tf.nn.static_rnn(cell=self.encoder, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32)
        return final_state

END_SYMBOL = '<eos>'
GO_SYMBOL = '<sos>'

# NMT decoder class, may like to generalize to general decoder, with different vocab etc
class NMTDecoder(tf.keras.Model):
    def __init__(self, V, word_dim, hidden_size, cell_type, time_major=True):
        super(NMTDecoder, self).__init__()
        self.word_embedding = Embedding(V, word_dim)
        self.cell_type = cell_type
        try:
            self.decoder = dict_rnn_cells[self.cell_type](num_units=hidden_size)
        except:
            assert(False)
        self.output_layer = tf.keras.layers.Dense(units=V)
        # TODO - now, SOS will be held fixed for all practical purposes. Do we want this?
        self.start_tok = self.word_embedding(tf.convert_to_tensor(np.full((args.bsize, 1), 3)))
        self.time_major = time_major

    # start_tok will be the sos sequences
    def call(self, encoder_state, mode, datum, max_iter):
        # During training, pass in the sos-starting input!
        if(mode == 'train'):
            # Do a language model
            word_indices = datum[0]; num_words = datum[1]
            word_vectors = self.word_embedding(word_indices)
            word_vectors_time = tf.unstack(word_vectors, axis=1)
            rnn_outputs_time, _ = tf.nn.static_rnn(cell=self.decoder, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32,initial_state=encoder_state)
            rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
            logits = self.output_layer(rnn_outputs)
            return logits
        
        elif(mode == 'eval'):
            # Do inference, return the decoding
            out = self.start_tok
            words_predicted, word_logits = [], []
            state = encoder_state
            for i in range(max_iter):
                out, state = self.decoder(out, state)
                logits = self.output_layer(out)
                # TODO - Do we even need softmax here? This is INFERENCE!
                pred_word = tf.argmax(logits)
                words_predicted.append(pred_word)
                word_logits.append(logits)
        
        word_logits = tf.stack(word_logits, axis=1)
        words_predicted = tf.stack(words_predicted, axis=1)
        return word_predicted, word_logits

# MLP for NLI class
class NLIDecoder(tf.keras.Model):
    def __init__(self, drop_p, hidden_dim, num_classes=3):
        super(NLIDecoder, self).__init__()
        self.drop_p = drop_p
        self.layer_size = hidden_dim*4 # Concatenate four vectors
        self.num_classes = num_classes
        # TODO - add dropout
        # TODO - add seed?
        self.dropout = tf.keras.layers.Dropout(drop_p)
        self.mlp_in = tf.keras.layers.Dense(units=self.layer_size, activation='relu', kernel_initializer='random_normal', bias_initializer='uniform')
        self.mlp_out = tf.keras.layers.Dense(units=self.num_classes, kernel_initializer='random_normal')        
    def call(self, in_vector):
        l1_drop = self.dropout(in_vector)
        l1 = self.mlp_in(l1_drop)
        logits = self.mlp_out(l1)    
        return logits

#--------------------END DEFINITIONS--------------------#

# Vocabularies
en_vocab_table = lookup_ops.index_table_from_file(args.en_vocab_file)
de_vocab_table = lookup_ops.index_table_from_file(args.de_vocab_file)

en_vocab_size = en_vocab_table.size(); de_vocab_size = de_vocab_table.size()


# Encoders and decoders
multi_encoder = SharedEncoder(en_vocab_size, args.embed_dim, args.hidden_dim, args.rnn_cell)
nmt_decoder = NMTDecoder(en_vocab_size, args.embed_dim, args.hidden_dim, args.rnn_cell)
nli_decoder = NLIDecoder(args.dropout, args.hidden_dim)

nli_batch = data_feed.get_nli_data(args.nli_premise, args.nli_hypothesis, args.nli_class, args.en_vocab_file, args.nli_class_vocab, args.bsize)

# nmt_batch = data_feed.get_nmt_data(en_d_small, de_d_small, en_v, de_v, bsize)
# nmt_batch = get_nmt_data(en_d_small, de_d_small, en_v, de_v, bsize)

def loss_nli(encoder, decoder_nli, datum):
    u = encoder(datum[0], datum[1])
    v = encoder(datum[2], datum[3])
    pass_in = tf.concat((u, v, u-v, u*v), axis=1)
    logits = decoder_nli(pass_in)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[4])
    return tf.reduce_sum(loss)/ tf.cast(args.bsize,dtype=tf.float32)

# TODO - Optional - remove extra column of sentence lengths, it is redundant
def loss_nmt(encoder, decoder_nmt, data):
    encoder_state = encoder(data[0][0], data[0][1])
    # def call(self, encoder_state, mode, datum=None):
    logits = decoder_nmt(encoder_state, 'train', data[1], 300)
    mask = tf.sequence_mask(data[2][1], dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=data[2][0]) * mask
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(data[2][1]), dtype=tf.float32)    

nli_loss_grads = tfe.implicit_value_and_gradients(loss_nli)
nmt_loss_grads = tfe.implicit_value_and_gradients(loss_nmt)

opt = tf.train.AdamOptimizer(learning_rate=args.lr)


STATS_STEPS = 5

for epoch_num in range(args.num_epochs):
    batch_loss = []
    for step_num, datum in enumerate(nli_batch, start=1):
        loss_value, gradients = nli_loss_grads(multi_encoder, nli_decoder, datum)
        batch_loss.append(loss_value)
        
        if step_num % STATS_STEPS == 0:
            print(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))}')
            batch_loss = []
        opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())
    print(f'Epoch{epoch_num} Done!')

for epoch_num in range(args.num_epochs):
    batch_loss = []
    for step_num, datum in enumerate(nmt_batch, start=1):
        loss_value, gradients = nmt_loss_grads(multi_encoder, nmt_decoder, datum)
        batch_loss.append(loss_value)
        
        if step_num % STATS_STEPS == 0:
            print(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))}')
            batch_loss = []
        opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())
    print(f'Epoch{epoch_num} Done!')

