import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tensorflow.python.ops import lookup_ops

dict_rnn_cells = {
                    'gru': tf.nn.rnn_cell.GRUCell,
                    'lstm': tf.nn.rnn_cell.LSTMCell,
                    'vanilla': tf.nn.rnn_cell.BasicRNNCell
                 }


# Embedding model
class Embedding(tf.keras.Model):
    def __init__(self, V, d):
        super(Embedding, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
    
    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)

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
        self.start_tok = tf.squeeze(self.word_embedding(tf.convert_to_tensor(np.full((args.bsize, 1), 3))), axis=1)
        self.time_major = time_major

    # start_tok will be the sos sequences
    def call(self, encoder_state, mode, datum, max_iter):
        # During training, pass in the sos-starting input!
        # if(mode == 'train'):
        # Do a language model
        word_indices = datum[0]; num_words = datum[1]
        word_vectors = self.word_embedding(word_indices)
        if(mode == 'train'):
            word_vectors = tf.nn.dropout(word_vectors, keep_prob=1-args.dropout)
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        rnn_outputs_time, _ = tf.nn.static_rnn(cell=self.decoder, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32,initial_state=encoder_state)
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        if(mode == 'train'):
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1-args.dropout)
        logits = self.output_layer(rnn_outputs)
        return logits                

# :: Parsing decoder .Very similar to NMT decoder
class ParseDecoder(tf.keras.Model):
    def __init__(self, V, word_dim, hidden_size, cell_type, time_major=True):
        super(ParseDecoder, self).__init__()
        self.word_embedding = Embedding(V, word_dim)
        self.cell_type = cell_type
        try:
            self.decoder = dict_rnn_cells[self.cell_type](num_units=hidden_size)
        except:
            assert(False)
        self.output_layer = tf.keras.layers.Dense(units=V)
        # TODO - now, SOS will be held fixed for all practical purposes. Do we want this?
        # For the Parse Decoder, the sos is 19
        self.start_tok = tf.squeeze(self.word_embedding(tf.convert_to_tensor(np.full((args.bsize, 1), 19))), axis=1)
        self.time_major = time_major

    # start_tok will be the sos sequences
    def call(self, encoder_state, mode, datum, max_iter):
        # During training, pass in the sos-starting input!
        # if(mode == 'train'):
        # Do a language model
        word_indices = datum[0]; num_words = datum[1]
        word_vectors = self.word_embedding(word_indices)
        if(mode == 'train'):
            word_vectors = tf.nn.dropout(word_vectors, keep_prob=1-args.dropout)
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        rnn_outputs_time, _ = tf.nn.static_rnn(cell=self.decoder, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32,initial_state=encoder_state)
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        if(mode == 'train'):
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1-args.dropout)
        logits = self.output_layer(rnn_outputs)
        return logits

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
    def call(self, in_vector,mode):
        l1_drop = []
        if(mode == 'train'):
            l1_drop = self.dropout(in_vector)
        else:
            l1_drop = in_vector
        l1 = self.mlp_in(l1_drop)
        logits = self.mlp_out(l1)    
        return logits
