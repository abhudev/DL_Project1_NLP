import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.ops import lookup_ops
import argparse
import time
import sys
from itertools import groupby

# :: Number of threads ::
num_threads = 2


def get_sentence_data(sen, en_vocab, bsize, need_eos, unk='<unk>',eos='<eos>',sos='<sos>'):
    """ Generic sentence batcher. """
    dataset = tf.data.TextLineDataset(sen)
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)

    dataset = dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    dataset = dataset.map(lambda words: en_vocab_table.lookup(words), num_parallel_calls=num_threads)
    dataset = dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    # dataset = tf.data.Dataset.zip((dataset, orig_dataset))    
    dataset = dataset.padded_batch(batch_size=bsize, padded_shapes=( [None], [] ) )
    return dataset


def get_nmt_data(en_data, de_data, en_vocab, de_vocab, bsize, mode, unk='<unk>',eos='<eos>', sos='<sos>'):
    """ NMT batcher
        Return original, translated, and translated+shifted sentence.
    """
    en_dataset = tf.data.TextLineDataset(en_data)
    de_dataset = tf.data.TextLineDataset(de_data)

    # : Default is <unk>
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)
    de_vocab_table = lookup_ops.index_table_from_file(de_vocab, default_value=1)

    # : Append EOS and make separate column for words :
    en_dataset = en_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    en_dataset = en_dataset.map(lambda words: tf.concat([words, [eos]], axis=0) , num_parallel_calls=num_threads)
    en_dataset = en_dataset.map(lambda words: en_vocab_table.lookup(words), num_parallel_calls=num_threads)
    en_dataset = en_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)

    # : Append SOS and EOS to two copies of the sentence, add extra columns for sentence length :
    de_dataset = de_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: words[:45], num_parallel_calls=num_threads)
    de_dataset_sos = de_dataset.map(lambda words: tf.concat([[sos], words], axis=0) , num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: tf.concat([words, [eos]], axis=0) , num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: de_vocab_table.lookup(words), num_parallel_calls=num_threads)
    de_dataset_sos = de_dataset_sos.map(lambda words: de_vocab_table.lookup(words), num_parallel_calls=num_threads)
    de_dataset = de_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    de_dataset_sos = de_dataset_sos.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    
    # : Zip translations line-by-line :
    en_de_dataset = tf.data.Dataset.zip((en_dataset, de_dataset_sos, de_dataset))    
    if(mode == 'train'):   
        en_de_dataset = en_de_dataset.shuffle(buffer_size=1000, seed=42)
    en_de_dataset = en_de_dataset.padded_batch(batch_size=bsize, padded_shapes=(([None], []), ([None], []), ([None], [])))
    en_de_dataset = en_de_dataset.prefetch(1)    
    return (en_de_dataset)    


def get_nli_data(nli_premise, nli_hypothesis, nli_classes, en_vocab, class_vocab, bsize, mode, unk='<unk>',eos='<eos>'):
    """ NLI Batcher
        Return Premise, Hypothesis sentence + class
        Also return sentence lengths
    """
    nli_premise = tf.data.TextLineDataset(nli_premise)
    nli_hypothesis = tf.data.TextLineDataset(nli_hypothesis)
    nli_classes = tf.data.TextLineDataset(nli_classes)
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)
    class_vocab_table = lookup_ops.index_table_from_file(class_vocab, default_value=0)
    
    # : Append EOS and make columns for number of words :
    nli_premise = nli_premise.map(lambda sentence: (tf.string_split([sentence]).values), num_parallel_calls=num_threads)
    nli_premise = nli_premise.map(lambda words: en_vocab_table.lookup(tf.concat([words, [eos]], axis=0)) , num_parallel_calls=num_threads)
    nli_premise_wrds = nli_premise.map(lambda words: tf.size(words) , num_parallel_calls=num_threads)

    nli_hypothesis = nli_hypothesis.map(lambda sentence: (tf.string_split([sentence]).values), num_parallel_calls=num_threads)
    nli_hypothesis = nli_hypothesis.map(lambda words: en_vocab_table.lookup(tf.concat([words, [eos]], axis=0)) , num_parallel_calls=num_threads)
    nli_hypothesis_wrds = nli_hypothesis.map(lambda words: tf.size(words), num_parallel_calls=num_threads)

    nli_classes = nli_classes.map(lambda sentence: class_vocab_table.lookup(tf.string_split([sentence]).values)[0], num_parallel_calls=num_threads)
    
    nli_dataset = tf.data.Dataset.zip((nli_premise, nli_premise_wrds, nli_hypothesis, nli_hypothesis_wrds, nli_classes))    
    if(mode == 'train'):
        nli_dataset = nli_dataset.shuffle(buffer_size=1000, seed=42)
    nli_dataset = nli_dataset.padded_batch(batch_size=bsize, padded_shapes=( (([None]),([]),([None]),([]),([])) ) )
    nli_dataset = nli_dataset.prefetch(1)
    
    return(nli_dataset)
    

def get_parse_data(sent_file, parse_file, en_vocab, parse_vocab, bsize, mode, unk='<unk>', eos='<eos>'):
    """ Parse data batcher
        Return sentence + linearized parse tree.        
    """
    sent_dataset = tf.data.TextLineDataset(sent_file)
    parse_dataset = tf.data.TextLineDataset(parse_file)

    # : Default value of unkown tokens
    en_vocab_table = lookup_ops.index_table_from_file(en_vocab, default_value=1)
    parse_vocab_table = lookup_ops.index_table_from_file(parse_vocab, default_value=1)    

    # : Append EOS and make column for number of words :
    sent_dataset = sent_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    sent_dataset = sent_dataset.map(lambda words: tf.concat([words, [eos]], axis=0) , num_parallel_calls=num_threads)
    sent_dataset = sent_dataset.map(lambda words: en_vocab_table.lookup(words), num_parallel_calls=num_threads)
    sent_dataset = sent_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)    

    # : Make shifted pairs and make column for number of words :
    parse_dataset = parse_dataset.map(lambda sentence: tf.string_split([sentence]).values, num_parallel_calls=num_threads)
    parse_dataset_start = parse_dataset.map(lambda words: words[:-1])    
    parse_dataset = parse_dataset.map(lambda words: parse_vocab_table.lookup(words[1:]), num_parallel_calls=num_threads)    
    parse_dataset_start = parse_dataset_start.map(lambda words: parse_vocab_table.lookup(words), num_parallel_calls=num_threads)
    parse_dataset = parse_dataset.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    parse_dataset_start = parse_dataset_start.map(lambda words: (words, tf.size(words)), num_parallel_calls=num_threads)
    
    # : Zip the two datasets with line-by-line parses :
    sen_parse_dataset = tf.data.Dataset.zip((sent_dataset, parse_dataset_start, parse_dataset))
    if(mode == 'train'):  
        sen_parse_dataset = sen_parse_dataset.shuffle(buffer_size=1000, seed=42)
    sen_parse_dataset = sen_parse_dataset.padded_batch(batch_size=bsize, padded_shapes=(([None], []), ([None], []), ([None], []) ))
    sen_parse_dataset = sen_parse_dataset.prefetch(1)
    
    return (sen_parse_dataset)