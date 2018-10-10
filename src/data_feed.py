import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.ops import lookup_ops
import argparse
import time
import sys
from itertools import groupby

tf.enable_eager_execution()

num_threads = 6

# Return iterable to produce batches for NMT data
def get_nmt_data(en_data, de_data, en_vocab, de_vocab, bsize, unk='<unk>',eos='<eos>', sos='<sos>'):
    en_dataset = tf.data.TextLineDataset(en_data)
    de_dataset = tf.data.TextLineDataset(de_data)

    # Make the default value 1, for UNK!!!
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)
    de_vocab_table = lookup_ops.index_table_from_file(de_vocab, default_value=1)

    # TODO - explore - storing the length of the sentence?

    en_dataset = en_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    en_dataset = en_dataset.map(lambda words: tf.concat([words, [eos]], axis=0) , num_parallel_calls=num_threads)
    en_dataset = en_dataset.map(lambda words: en_vocab_table.lookup(words), num_parallel_calls=num_threads)
    en_dataset = en_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    # en_dataset_iter = iter(en_dataset)
    de_dataset = de_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    de_dataset_sos = de_dataset.map(lambda words: tf.concat([[sos], words], axis=0) , num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: tf.concat([words, [eos]], axis=0) , num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: de_vocab_table.lookup(words), num_parallel_calls=num_threads)
    de_dataset_sos = de_dataset_sos.map(lambda words: de_vocab_table.lookup(words), num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    de_dataset_sos = de_dataset_sos.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    # de_dataset_iter = iter(de_dataset)
    # Use this to zip the two datasets with line-by-line translations
    en_de_dataset = tf.data.Dataset.zip((en_dataset, de_dataset_sos, de_dataset))    
    en_de_batch = en_de_dataset.padded_batch(batch_size=bsize, padded_shapes=(([None], []), ([None], []), ([None], [])))
    en_de_batch = en_de_batch.prefetch(1)
    # return iter(en_de_batch)
    return (en_de_batch)
    # return en_de_dataset

#   
# bat = get_nmt_data(en_d, de_d, en_v, de_v, bsize)

# TODO - preprocess
def get_nli_data(nli_premise, nli_hypothesis, nli_classes, en_vocab, class_vocab, bsize,unk='<unk>',eos='<eos>'):
    nli_premise = tf.data.TextLineDataset(nli_premise)
    nli_hypothesis = tf.data.TextLineDataset(nli_hypothesis)
    nli_classes = tf.data.TextLineDataset(nli_classes)
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)
    class_vocab_table = lookup_ops.index_table_from_file(class_vocab, default_value=0)
    # TODO - explore - storing the length of the sentence?
    # Get LABELS!!
    nli_premise = nli_premise.map(lambda sentence: (tf.string_split([sentence]).values), num_parallel_calls=num_threads)
    nli_premise = nli_premise.map(lambda words: en_vocab_table.lookup(tf.concat([words, [eos]], axis=0)) , num_parallel_calls=num_threads)
    nli_premise_wrds = nli_premise.map(lambda words: tf.size(words) , num_parallel_calls=num_threads)
    # nli_hypothesis = nli_hypothesis.map(lambda sentence: en_vocab_table.lookup(tf.string_split([sentence]).values), num_parallel_calls=num_threads)
    nli_hypothesis = nli_hypothesis.map(lambda sentence: (tf.string_split([sentence]).values), num_parallel_calls=num_threads)
    nli_hypothesis = nli_hypothesis.map(lambda words: en_vocab_table.lookup(tf.concat([words, [eos]], axis=0)) , num_parallel_calls=num_threads)
    nli_hypothesis_wrds = nli_hypothesis.map(lambda words: tf.size(words), num_parallel_calls=num_threads)
    nli_classes = nli_classes.map(lambda sentence: class_vocab_table.lookup(tf.string_split([sentence]).values)[0], num_parallel_calls=num_threads)
    nli_dataset = tf.data.Dataset.zip((nli_premise, nli_premise_wrds, nli_hypothesis, nli_hypothesis_wrds, nli_classes))
    # nli_dataset = nli_dataset.map(lambda tup: tup[0], num_parallel_calls=num_threads)
    nli_batch = nli_dataset.padded_batch(batch_size=bsize, padded_shapes=( (([None]),([]),([None]),([]),([])) ) )
    nli_batch = nli_batch.prefetch(1)
    # return iter(nli_batch)
    return(nli_batch)
    # return nli_dataset

# bat = get_nli_data(nli_p, nli_h, nli_c, en_v, c_v, bsize)
# bat = get_nli_data(nli_p_small, nli_h_small, nli_c_small, en_v, c_v, bsize)
# nli_batch = data_feed.get_nli_data(nli_p_small, nli_h_small, nli_c_small, args.en_vocab_file, args.nli_class_vocab, args.bsize)

# def get_nmt_data(en_data, de_data, en_vocab, de_vocab, bsize, unk='<unk>',eos='<eos>', sos='<sos>'):
def get_parse_data(sent_file, parse_file, en_vocab, parse_vocab, bsize, unk='<unk>', eos='<eos>'):
    sent_dataset = tf.data.TextLineDataset(sent_file)
    parse_dataset = tf.data.TextLineDataset(parse_file)

    # Make the default value 1, for UNK!!!
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)
    parse_vocab_table = lookup_ops.index_table_from_file(parse_vocab, default_value=1)

    # TODO - explore - storing the length of the sentence?

    sent_dataset = sent_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    sent_dataset = sent_dataset.map(lambda words: tf.concat([words, [eos]], axis=0) , num_parallel_calls=num_threads)
    sent_dataset = sent_dataset.map(lambda words: en_vocab_table.lookup(words), num_parallel_calls=num_threads)
    sent_dataset = sent_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    # en_dataset_iter = iter(en_dataset)

    parse_dataset = parse_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    parse_dataset_start = parse_dataset.map(lambda words: words[:-1])    
    parse_dataset = parse_dataset.map(lambda words: parse_vocab_table.lookup(words[1:]), num_parallel_calls=num_threads)    
    parse_dataset_start = parse_dataset_start.map(lambda words: parse_vocab_table.lookup(words), num_parallel_calls=num_threads)
    parse_dataset = parse_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    parse_dataset_start = parse_dataset_start.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    # de_dataset_iter = iter(de_dataset)
    # Use this to zip the two datasets with line-by-line translations
    sen_parse_dataset = tf.data.Dataset.zip((sent_dataset, parse_dataset_start, parse_dataset))    
    sen_parse_batch = sen_parse_dataset.padded_batch(batch_size=bsize, padded_shapes=(([None], []), ([None], []), ([None], []) ))
    sen_parse_batch = sen_parse_batch.prefetch(1)
    # return iter(en_de_batch)
    return (sen_parse_batch)
    # return en_de_dataset