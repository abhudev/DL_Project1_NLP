import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--nli_file', type=str, default='../PROJECT_data/NLI/allnli.train.txt.clean.noblank')
parser.add_argument('--out_premise', type=str, default='../PROJECT_data/NLI/allnli_premise.txt')
parser.add_argument('--out_hypothesis', type=str, default='../PROJECT_data/NLI/allnli_hypothesis.txt')
parser.add_argument('--out_classes', type=str, default='../PROJECT_data/NLI/allnli_train_classes.txt')
args = parser.parse_args()

start = time.time()

classes = {}
classes['neutral'] = 0; classes['contradiction'] = 1; classes['entailment'] = 2

with open(args.nli_file, encoding="utf-8") as fin, open(args.out_premise, "w", encoding="utf-8") as foutp, open(args.out_hypothesis, "w", encoding="utf-8") as fouth, open(args.out_classes, "w", encoding="utf-8") as foutc:
    for num, rline in enumerate(fin):
        line = rline.strip()
        s1 = line.find("\t")
        line_premise = line[:s1]; line = line[s1+1:]
        s2 = line.find("\t")
        line_hypothesis = line[:s2]
        line_class = line.split()[-1]
        
        # list_pos = [" ."+x if (x == "\t" and line[i-1]!=".") else x for i, x in enumerate(line)]
        # new_line = "".join(list_pos)
        
        foutp.write(line_premise+"\n")
        fouth.write(line_hypothesis+"\n")
        foutc.write(line_class+"\n")
        # foutc.write(str(classes[line_class])+"\n")
        if((num+1)%10000 == 0):
            sys.stdout.write(f"\rLine {num+1}")
            sys.stdout.flush()
    end = time.time()
    total = end-start
    print(f"\nTime taken = {total} seconds")


