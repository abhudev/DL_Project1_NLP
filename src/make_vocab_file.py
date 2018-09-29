#Let us define some handles for input, output and vocab files
input_file = '../data/wiki.1M.txt'
output_file = '../data/wiki.1M.txt.tokenized'
vocab_file = '../data/vocab.txt'

from nltk.tokenize import word_tokenize
from collections import OrderedDict
import argparse
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--nli_data', type=str, default='../PROJECT_data/NLI/allnli.train.txt.clean.noblank')
parser.add_argument('--nmt_en', type=str, default='../PROJECT_data/NMT/nmt.de-en.en.tok')
parser.add_argument('--nmt_de', type=str, default='../PROJECT_data/NMT/nmt.de-en.de.tok')
parser.add_argument('--en_vocab_file',type=str, default='../PROJECT_data/en_vocab.txt')
parser.add_argument('--de_vocab_file',type=str, default='../PROJECT_data/de_vocab.txt')
parser.add_argument('--vocab_size', type=str, default=30000)
args = parser.parse_args()

#TODO: Fill in the method below:
def convert_to_tokens(input_file, output_file):
    with open(input_file) as fr,open(output_file, 'w') as fw:
        for index, sentence in enumerate(fr):
            words = word_tokenize(sentence.strip().lower())
            fw.write(f"{' '.join(words)}\n")
            if index % 100000 == 0:
                print(index)

#This should take about 2 minutes. Think that is long! Think again, you processed 1M sentences in under 2 minutes!
# start = time.time()
# convert_to_tokens(input_file, output_file)
# print(f'Time Taken: {time.time()-start}s')

#TODO: Count words and return words in a sorted order
# Pass in LIST of files, and tell which is which - NMT, allnli or others
def count_words_english(nli_data, nmt_en):
    counter = dict()
    i = 0
    for sentence in open(nli_data,encoding='utf-8'):
        words = sentence.strip().split()
        i += 1
        if(i%10000 == 0):            
            sys.stdout.write("\rSentence {}".format(i))
            sys.stdout.flush()
        for word in words[:-1]: #Last word is tag
            counter[word] = counter.get(word, 0) + 1    
    for sentence in open(nmt_en,encoding='utf-8'):
        i += 1
        if(i%10000 == 0):            
            sys.stdout.write("\rSentence {}".format(i))
            sys.stdout.flush()
        words = sentence.strip().split()
        for word in words: #Last word is tag
            counter[word] = counter.get(word, 0) + 1
    print("")
    return sorted(counter.items(), key=lambda pair:pair[1], reverse=True)


def count_words_german(nmt_de):
    counter = dict()
    i = 0
    for sentence in open(nmt_de,encoding='utf-8'):
        i += 1
        if(i%1000 == 0):
            sys.stdout.write("\rSentence {}".format(i))
            sys.stdout.flush()
        words = sentence.strip().split()
        for word in words: #Last word is tag
            counter[word] = counter.get(word, 0) + 1
    print("")
    return sorted(counter.items(), key=lambda pair:pair[1], reverse=True)


print("Processing english...")
start = time.time()
en_word_counts = count_words_english(args.nli_data, args.nmt_en)
end = time.time()
print(f"Done in {end-start}")
print(en_word_counts[:10])
print("Processing german...")
start = time.time()
de_word_counts = count_words_german(args.nmt_de)
end = time.time()
print(f"Done in {end-start}")
print(de_word_counts[:10])

# exit()

# ### P7: Assign word index


#TODO: Create a word->integer mapping
# Discard words which have count less than min freq
# Assign index 0 to `<unk>`
# Make it different. Make unk - 1, eos - 2. 0 - reserved for pad
def build_vocab(word_counts, size):
    vocab = OrderedDict()
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    vocab['<eos>'] = 2
    # num_words = 0
    a = 0
    for word, freq in word_counts:        
        cur_size = len(vocab)
        vocab[word] = cur_size + 1
        if(cur_size >= size):
            print(f"Lowest frequency = {freq}")
            return vocab
    return vocab



en_vocab = build_vocab(en_word_counts, args.vocab_size)
de_vocab = build_vocab(de_word_counts, args.vocab_size)
print(f'V[<unk>]: {en_vocab["<unk>"]} V["learning"]: {en_vocab["learning"]}')
print(f'V2[<unk>]: {de_vocab["<unk>"]} V2["learning"]: {de_vocab["ist"]}')

#Return a list of word indexes for all words in sentence
def assign_word_indexes(sentence, vocab):
    return [vocab[word] if word in vocab else vocab['<unk>'] for word in sentence.split()]



word_indexes = assign_word_indexes('i am learning to build vocab', en_vocab)


print(word_indexes)
print(len(en_vocab))


# ### P8: Write Vocabulary file
# * Write each word in vocabulary to a new line



def write_vocab_file(vocab_file, vocab):
    with open(vocab_file, 'w',encoding='utf-8') as fw:
        for word in vocab:
            fw.write(f'{word}\n')



write_vocab_file(args.en_vocab_file, en_vocab)
write_vocab_file(args.de_vocab_file, de_vocab)


# How many lines in our vocab file?
