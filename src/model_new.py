import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
tf.set_random_seed(42)
import numpy as np
from tensorflow.python.ops import lookup_ops
from collections import OrderedDict
import argparse
import data_feed
from tensorflow.python.layers.core import Dense
import os
import time

logging = tf.logging
logging.set_verbosity(logging.INFO)
def log_msg(msg):
    logging.info(f'{time.ctime()}: {msg}')


parser = argparse.ArgumentParser()

# :: NLI train data ::
parser.add_argument('--nli_data', type=str, default='../PROJECT_data/NLI/allnli_train.txt')
parser.add_argument('--nli_premise', type=str, default='../PROJECT_data/NLI/allnli_train_premise.txt')
parser.add_argument('--nli_hypothesis', type=str, default='../PROJECT_data/NLI/allnli_train_hypothesis.txt')
parser.add_argument('--nli_class', type=str, default = '../PROJECT_data/NLI/allnli_train_classes.txt')
# :: NLI dev data ::
parser.add_argument('--nli_data_dev', type=str, default='../PROJECT_data/NLI/allnli_dev.txt')
parser.add_argument('--nli_premise_dev', type=str, default='../PROJECT_data/NLI/allnli_dev_premise.txt')
parser.add_argument('--nli_hypothesis_dev', type=str, default='../PROJECT_data/NLI/allnli_dev_hypothesis.txt')
parser.add_argument('--nli_class_dev', type=str, default = '../PROJECT_data/NLI/allnli_dev_classes.txt')
# :: NLI test data ::
parser.add_argument('--nli_data_test', type=str, default='../PROJECT_data/NLI/allnli_test.txt')
parser.add_argument('--nli_premise_test', type=str, default='../PROJECT_data/NLI/allnli_test_premise.txt')
parser.add_argument('--nli_hypothesis_test', type=str, default='../PROJECT_data/NLI/allnli_test_hypothesis.txt')
parser.add_argument('--nli_class_test', type=str, default = '../PROJECT_data/NLI/allnli_test_classes.txt')

# :: NMT train data ::
parser.add_argument('--nmt_en', type=str, default='../PROJECT_data/NMT/en_train_nmt.txt')
parser.add_argument('--nmt_de', type=str, default='../PROJECT_data/NMT/de_train_nmt.txt')
# :: NMT dev data ::
parser.add_argument('--nmt_en_dev', type=str, default='../PROJECT_data/NMT/en_dev_nmt.txt')
parser.add_argument('--nmt_de_dev', type=str, default='../PROJECT_data/NMT/de_dev_nmt.txt')
# :: NMT test data ::
parser.add_argument('--nmt_en_test', type=str, default='../PROJECT_data/NMT/en_test_nmt.txt')
parser.add_argument('--nmt_de_test', type=str, default='../PROJECT_data/NMT/de_test_nmt.txt')

# :: Checkpoint directory and name ::
parser.add_argument('--ckpt_dir', type=str, default='../Checkpoints')
parser.add_argument('--ckpt_file', type=str, default='ckpt')
parser.add_argument('--ckpt_nli', type=str, default='../ckpt_nli')
parser.add_argument('--ckpt_parse', type=str, default='../ckpt_parse')
parser.add_argument('--ckpt_nmt', type=str, default='../ckpt_nmt')
parser.add_argument('--ckpt_dir_regular', type=str, default='../Checkpoints_regular')
parser.add_argument('--ckpt_nli_regular', type=str, default='../ckpt_nli_regular')
parser.add_argument('--ckpt_nmt_regular', type=str, default='../ckpt_nmt_regular')
parser.add_argument('--ckpt_parse_regular', type=str, default='../ckpt_parse_regular')

# :: Parse train data ::
parser.add_argument('--sent_data', type=str, default='../PROJECT_data/Parsing/Berkely/trunc_train_lower_fixed.txt')
parser.add_argument('--linear_data', type=str, default='../PROJECT_data/Parsing/Berkely/parse_train_lin2_fixed.txt')
# :: Parse dev data ::
parser.add_argument('--sent_data_dev', type=str, default='../PROJECT_data/Parsing/Berkely/trunc_dev_lower_fixed.txt')
parser.add_argument('--linear_data_dev', type=str, default='../PROJECT_data/Parsing/Berkely/parse_dev_lin2_fixed.txt')
# :: Parse test data ::
parser.add_argument('--sent_data_test', type=str, default='../PROJECT_data/Parsing/Berkely/trunc_test_lower_fixed.txt')
parser.add_argument('--linear_data_test', type=str, default='../PROJECT_data/Parsing/Berkely/parse_test_lin2_fixed.txt')

# :: Vocabularies for text and classes ::
parser.add_argument('--en_vocab_file',type=str, default='../PROJECT_data/en_vocab.txt')
parser.add_argument('--de_vocab_file',type=str, default='../PROJECT_data/de_vocab.txt')
parser.add_argument('--nli_class_vocab', type=str, default='../PROJECT_data/NLI/nli_class_vocab.txt')
parser.add_argument('--parse_vocab_file', type=str, default='../PROJECT_data/Parsing/Berkely/parse_vocab.txt')

# :: Architecture parameters ::
parser.add_argument('--rnn_cell', type=str, default='gru')
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--bsize', type=int, default=20)
parser.add_argument('--dev_bsize', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--start_task', type=str, default='nmt')
# parser.add_argument('--')
# parser.add_argument('--vocab_size', type=str, default=30000)
args = parser.parse_args()


# tf.set_random_seed(42)

#--------------------BEGIN DEFINITIONS--------------------#

# Embedding model
class Embedding(tf.keras.Model):
    def __init__(self, V, d):
        super(Embedding, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))
    
    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)

# Different kinds of RNN cells to be used
# dict_fast_rnn_cells = {'GPU': 
#                         {'gru': tf.contrib.cudnn_rnn.CudnnGRU, 
#                          'lstm': tf.contrib.cudnn_rnn.CudnnLSTM, 
#                          'vanilla':  tf.contrib.cudnn_rnn.CudnnRNNTanh}, 

#                   'CPU': 
#                         {'gru': tf.contrib.rnn.GRUBlockCellV2, 
#                          'lstm': tf.contrib.rnn.LSTMBlockCell, 
#                          'vanilla': tf.nn.rnn_cell.BasicRNNCell}
#                  }

# Different kinds of RNN cells
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
        # word_vectors = self.word_embedding(word_indices)
        # word_vectors_time = tf.unstack(word_vectors, axis=1)
        # word_vectors_time = tf.unstack(self.word_embedding(word_indices), axis=1)
        _, final_state = tf.nn.static_rnn(cell=self.encoder, inputs=tf.unstack(self.word_embedding(word_indices), axis=1), sequence_length=num_words, dtype=tf.float32)
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
        self.start_tok = tf.squeeze(self.word_embedding(tf.convert_to_tensor(np.full((args.bsize, 1), 3))), axis=1)
        self.time_major = time_major

    # start_tok will be the sos sequences
    def call(self, encoder_state, mode, datum, max_iter):
        # During training, pass in the sos-starting input!
        # if(mode == 'train'):
        # Do a language model
        word_indices = datum[0]; num_words = datum[1]
        word_vectors = self.word_embedding(word_indices)
        word_indices = None
        if(mode == 'train'):
            word_vectors = tf.nn.dropout(word_vectors, keep_prob=1-args.dropout)
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        word_vectors = None
        rnn_outputs_time, _ = tf.nn.static_rnn(cell=self.decoder, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32,initial_state=encoder_state)
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        rnn_outputs_time = None
        if(mode == 'train'):
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1-args.dropout)
        logits = self.output_layer(rnn_outputs)
        rnn_outputs = None
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
        word_indices = None
        if(mode == 'train'):
            word_vectors = tf.nn.dropout(word_vectors, keep_prob=1-args.dropout)
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        word_vectors = None
        rnn_outputs_time, _ = tf.nn.static_rnn(cell=self.decoder, inputs=word_vectors_time, sequence_length=num_words, dtype=tf.float32,initial_state=encoder_state)
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        rnn_outputs_time = None
        if(mode == 'train'):
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=1-args.dropout)
        logits = self.output_layer(rnn_outputs)
        rnn_outputs = None
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
        l1_drop = None
        logits = self.mlp_out(l1)
        l1 = None
        return logits

#--------------------END DEFINITIONS--------------------#

# Vocabularies
en_vocab_table = lookup_ops.index_table_from_file(args.en_vocab_file)
de_vocab_table = lookup_ops.index_table_from_file(args.de_vocab_file)
parse_vocab_table = lookup_ops.index_table_from_file(args.parse_vocab_file)

en_vocab_size = en_vocab_table.size(); de_vocab_size = de_vocab_table.size(); parse_vocab_size = parse_vocab_table.size()


# Encoders and decoders
multi_encoder = SharedEncoder(en_vocab_size, args.embed_dim, args.hidden_dim, args.rnn_cell)
nmt_decoder = NMTDecoder(en_vocab_size, args.embed_dim, args.hidden_dim, args.rnn_cell)
nli_decoder = NLIDecoder(args.dropout, args.hidden_dim)
parse_decoder = ParseDecoder(parse_vocab_size, args.embed_dim, args.hidden_dim, args.rnn_cell)

# nmt_batch = data_feed.get_nmt_data(en_d_small, de_d_small, en_v, de_v, bsize)
# nmt_batch = get_nmt_data(en_d_small, de_d_small, en_v, de_v, bsize)


# Trying to reduce memory usage on gpu...
def loss_nli(encoder, decoder_nli, datum, mode):
    u = encoder(datum[0], datum[1])
    v = encoder(datum[2], datum[3])
    pass_in = tf.concat((u, v, u-v, u*v), axis=1)
    logits = decoder_nli(pass_in, mode)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[4])
    # if(mode == 'train'):
    return tf.reduce_sum(loss)/ tf.cast(args.bsize,dtype=tf.float32)
    u, v, pass_in, logits = None, None, None, None
    # return tf.reduce_sum(loss)/ tf.cast(args.dev_bsize,dtype=tf.float32)

# TODO - Optional - remove extra column of sentence lengths, it is redundant
def loss_nmt(encoder, decoder_nmt, data, mode):
    encoder_state = encoder(data[0][0], data[0][1])
    # def call(self, encoder_state, mode, datum=None):
    logits = decoder_nmt(encoder_state, mode, data[1], 300)
    mask = tf.sequence_mask(data[2][1], dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=data[2][0]) * mask
    encoder_state, logits, mask = None, None, None
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(data[2][1]), dtype=tf.float32)    

def loss_parse(encoder, decoder_parse, data, mode):
    encoder_state = encoder(data[0][0], data[0][1])
    # def call(self, encoder_state, mode, datum=None):
    logits = decoder_parse(encoder_state, mode, data[1], 500)
    mask = tf.sequence_mask(data[2][1], dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=data[2][0]) * mask
    encoder_state, logits, mask = None, None, None
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(data[2][1]), dtype=tf.float32)    

nli_loss_grads = tfe.implicit_value_and_gradients(loss_nli)
nmt_loss_grads = tfe.implicit_value_and_gradients(loss_nmt)
parse_loss_grads = tfe.implicit_value_and_gradients(loss_parse)

# :: Perplexity ::
def compute_ppl(encoder_model, decoder_model, dataset, task):
    total_loss = 0.
    total_words = 0
    
    loss_fun = []    
    if(task == 'nli'):
        loss_fun = loss_nli
    elif(task == 'nmt'):
        loss_fun = loss_nmt
    elif(task == 'parse'):
        loss_fun = loss_parse

    max_runs =  5000

    for batch_num, datum in enumerate(dataset):
        num_words = 0
        if(task == 'nli'):
            num_words = int( args.bsize )
        elif(task == 'nmt'):
            num_words = int(tf.reduce_sum(datum[1][1]))            
        elif(task == 'parse'):
            num_words = int(tf.reduce_sum(datum[1][1]))

        avg_loss = loss_fun(encoder_model, decoder_model, datum, 'eval')
        total_loss += avg_loss * num_words
        total_words += num_words                

        if batch_num % 500 == 0:
            log_msg(f'{task} ppl Done batch: {batch_num}')
        if(batch_num >= max_runs):
            break
    
    loss = total_loss / float(num_words)
    # if(task == 'nli' or task == 'parse'):
    return loss
    # return np.exp(loss)


opt = tf.train.AdamOptimizer(learning_rate=args.lr)

STATS_STEPS = 100
EVAL_STEPS = 500

# :: Load datasets ::
nli_batch = data_feed.get_nli_data(args.nli_premise, args.nli_hypothesis, args.nli_class, args.en_vocab_file, args.nli_class_vocab, args.bsize, 'train')
nmt_batch = data_feed.get_nmt_data(args.nmt_en, args.nmt_de, args.en_vocab_file, args.de_vocab_file, args.bsize, 'train')
parse_batch = data_feed.get_parse_data(args.sent_data, args.linear_data, args.en_vocab_file, args.parse_vocab_file, args.bsize, 'train')

nli_batch_dev = data_feed.get_nli_data(args.nli_premise_dev, args.nli_hypothesis_dev, args.nli_class_dev, args.en_vocab_file, args.nli_class_vocab, args.dev_bsize, 'eval')
nmt_batch_dev = data_feed.get_nmt_data(args.nmt_en_dev, args.nmt_de_dev, args.en_vocab_file, args.de_vocab_file, args.dev_bsize, 'eval')
parse_batch_dev = data_feed.get_parse_data(args.sent_data_dev, args.linear_data_dev, args.en_vocab_file, args.parse_vocab_file, args.dev_bsize, 'eval')

# nli_batch_test = data_feed.get_nli_data(args.nli_premise_test, args.nli_hypothesis_test, args.nli_class_test, args.en_vocab_file, args.nli_class_vocab, args.dev_bsize, 'test')
# nmt_batch_test = data_feed.get_nmt_data(args.nmt_en_test, args.nmt_de_test, args.en_vocab_file, args.de_vocab_file, args.dev_bsize, 'test')
# parse_batch_test = data_feed.get_parse_data(args.sent_data_test, args.linear_data_test, args.en_vocab_file, args.parse_vocab_file, args.dev_bsize, 'test')

ckpt_prefix = os.path.join(args.ckpt_dir, args.ckpt_file)
reg_ckpt = os.path.join(args.ckpt_dir_regular, args.ckpt_file)

nli_ckpt_prefix = os.path.join(args.ckpt_nli, args.ckpt_file)
nli_reg_ckpt = os.path.join(args.ckpt_nli_regular, args.ckpt_file)

nmt_ckpt_prefix = os.path.join(args.ckpt_nmt, args.ckpt_file)
nmt_reg_ckpt = os.path.join(args.ckpt_nmt_regular, args.ckpt_file)

parse_ckpt_prefix = os.path.join(args.ckpt_parse, args.ckpt_file)
parse_reg_ckpt = os.path.join(args.ckpt_parse_regular, args.ckpt_file)

saver = tfe.Checkpoint(optimizer=opt, model=multi_encoder, optimizer_step=tf.train.get_or_create_global_step())
saver_nli = tfe.Checkpoint(optimizer=opt, model=nli_decoder, optimizer_step=tf.train.get_or_create_global_step())
saver_nmt = tfe.Checkpoint(optimizer=opt, model=nmt_decoder, optimizer_step=tf.train.get_or_create_global_step())
saver_parse = tfe.Checkpoint(optimizer=opt, model=parse_decoder, optimizer_step=tf.train.get_or_create_global_step())

saver.restore(tf.train.latest_checkpoint(ckpt_prefix))
saver_nli.restore(tf.train.latest_checkpoint(nli_ckpt_prefix))
saver_nmt.restore(tf.train.latest_checkpoint(nmt_ckpt_prefix))
saver_parse.restore(tf.train.latest_checkpoint(parse_ckpt_prefix))

valid_ppl_nli = compute_ppl(multi_encoder, nli_decoder, nli_batch_dev,'nli')
valid_ppl_nmt = compute_ppl(multi_encoder, nmt_decoder, nmt_batch_dev,'nmt')
valid_ppl_parse = compute_ppl(multi_encoder, parse_decoder, parse_batch_dev,'parse')

# valid_ppl_nli = 10000000
# valid_ppl_nmt = 10000000
# valid_ppl_parse = 10000000

log_msg(f'Start :Dev ppl for NLI: {valid_ppl_nli}')
log_msg(f'Start :Dev ppl for NMT: {valid_ppl_nmt}')
log_msg(f'Start :Dev ppl for parse: {valid_ppl_parse}')


# regular_save = tfe.Checkpoint(optimizer=opt, model=multi_encoder, optimizer_step=tf.train.get_or_create_global_step())
# with tf.device("/gpu:0"):
start_reg = time.time()
for epoch_num in range(args.num_epochs):
    # batch_loss = []        
    if(epoch_num > 0):
        saver.restore(tf.train.latest_checkpoint(ckpt_prefix))
        saver_nli.restore(tf.train.latest_checkpoint(nli_ckpt_prefix))
        saver_nmt.restore(tf.train.latest_checkpoint(nmt_ckpt_prefix))
        saver_parse.restore(tf.train.latest_checkpoint(parse_ckpt_prefix))
    log_msg(f"Begin Epoch {epoch_num} of NMT with restored model")
    start_reg = time.time()
    for step_num, datum in enumerate(nmt_batch, start=1):
        # log_msg(f"{int(datum[0][0].shape[1])}")
        # if(int(datum[0][0].shape[1]) > 45):
        #    continue
        with tf.device("/gpu:0"):
            loss_value, gradients = nmt_loss_grads(multi_encoder, nmt_decoder, datum, 'train')                        
            opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())

        if step_num % STATS_STEPS == 0:
            log_msg(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))}')
            batch_loss = []
    
        if step_num % 11560 == 0:
            with tf.device("/gpu:0"):
                ppl = compute_ppl(multi_encoder, nmt_decoder, nmt_batch_dev,'nmt')            
            #Save model!
            if ppl < valid_ppl_nmt:
                saver.save(ckpt_prefix)
                saver_nli.save(nli_ckpt_prefix)
                saver_nmt.save(nmt_ckpt_prefix)
                saver_parse.save(parse_ckpt_prefix)
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl} old: {valid_ppl_nmt} Model saved')
                valid_ppl_nmt = ppl
            else:
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl worse: {ppl} old: {valid_ppl_nmt}')
            
        if((time.time() - start_reg)/3600 >= 1.0):
            saver.save(reg_ckpt)
            saver_nli.save(nli_reg_ckpt)
            saver_nmt.save(nmt_reg_ckpt)
            saver_parse.save(parse_reg_ckpt)
            log_msg(f'Epoch: {epoch_num} Step: {step_num} Model regularly saved')
            start_reg = time.time()        

    saver_nli.restore(tf.train.latest_checkpoint(nli_ckpt_prefix))
    saver_nmt.restore(tf.train.latest_checkpoint(nmt_ckpt_prefix))
    saver_parse.restore(tf.train.latest_checkpoint(parse_ckpt_prefix))
    log_msg(f"Begin Epoch {epoch_num} of Parsing")
    start_reg = time.time()
    for step_num, datum in enumerate(parse_batch, start=1):

        # print("Data: ", time.time() - start)
        # start = time.time()
        with tf.device("/gpu:0"):
            loss_value, gradients = parse_loss_grads(multi_encoder, parse_decoder, datum, 'train')
            # t1 = time.time()
            # batch_loss.append(loss_value)                                
            opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())

        if step_num % STATS_STEPS == 0:
            log_msg(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))}')
            # batch_loss = []

        # t2 = time.time()

        # print("Gradients: ",t1 - start)
        # print("Apply grad: ", t2 - t1)

        if step_num % 1230 == 0:
            with tf.device("/gpu:0"):
                ppl = compute_ppl(multi_encoder, parse_decoder, parse_batch_dev,'parse')
            #Save model!
            if ppl < valid_ppl_parse:
                saver.save(ckpt_prefix)
                saver_nli.save(nli_ckpt_prefix)
                saver_nmt.save(nmt_ckpt_prefix)
                saver_parse.save(parse_ckpt_prefix)
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl} old: {valid_ppl_parse} Model saved')
                valid_ppl_parse = ppl
            else:
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl worse: {ppl} old: {valid_ppl_parse}')
        # start = time.time()
        if((time.time() - start_reg)/3600 >= 1.0):
            saver.save(reg_ckpt)
            saver_nli.save(nli_reg_ckpt)
            saver_nmt.save(nmt_reg_ckpt)
            saver_parse.save(parse_reg_ckpt)
            log_msg(f'Epoch: {epoch_num} Step: {step_num} Model regularly saved')
            start_reg = time.time()

    log_msg(f'Parsing epoch{epoch_num} Done!')

    # if(epoch_num > 0):
    saver.restore(tf.train.latest_checkpoint(ckpt_prefix))
    saver_nli.restore(tf.train.latest_checkpoint(nli_ckpt_prefix))
    saver_nmt.restore(tf.train.latest_checkpoint(nmt_ckpt_prefix))
    saver_parse.restore(tf.train.latest_checkpoint(parse_ckpt_prefix))
    log_msg(f"Begin Epoch {epoch_num} of NLI")
    start_reg = time.time()
    for step_num, datum in enumerate(nli_batch, start=1):
        
        with tf.device("/gpu:0"):
            loss_value, gradients = nli_loss_grads(multi_encoder, nli_decoder, datum, 'train')        
            opt.apply_gradients(gradients, global_step=tf.train.get_or_create_global_step())    

        # batch_loss.append(loss_value)            
        # batch_loss = []

        if step_num % STATS_STEPS == 0:            
            log_msg(f'Epoch: {epoch_num} Step: {step_num} Avg Loss: {np.average(np.asarray(loss_value))}')
        
        if step_num % 2530 == 0:
            with tf.device("/gpu:0"):
                ppl = compute_ppl(multi_encoder, nli_decoder, nli_batch_dev,'nli')            
            #Save model!
            if ppl < valid_ppl_nli:
                saver.save(ckpt_prefix)
                saver_nli.save(nli_ckpt_prefix)
                saver_nmt.save(nmt_ckpt_prefix)
                saver_parse.save(parse_ckpt_prefix)
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl} old: {valid_ppl_nli} Model saved')
                valid_ppl_nli = ppl
            else:
                log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl worse: {ppl} old: {valid_ppl_nli}')
        if((time.time() - start_reg)/3600 >= 1.0):
            saver.save(reg_ckpt)
            saver_nli.save(nli_reg_ckpt)
            saver_nmt.save(nmt_reg_ckpt)
            saver_parse.save(parse_reg_ckpt)
            log_msg(f'Epoch: {epoch_num} Step: {step_num} Model regularly saved')
            start_reg = time.time()
    # LOL BALLS
    # batch_loss = []

