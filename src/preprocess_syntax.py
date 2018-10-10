import argparse
import os
import sys
import time
# from itertools import zip

parser = argparse.ArgumentParser()
parser.add_argument('--parsed_file', type=str, default='../PROJECT_data/Parsing/Berkely/parsed_train_corpus.txt')
parser.add_argument('--sent_file', type=str, default='../PROJECT_data/Parsing/Berkely/train_head.txt')
parser.add_argument('--out_parse', type=str, default='../PROJECT_data/Parsing/Berkely/linearized_parse.txt')
# parser.add_argument()
parser.add_argument('--start_token', type=str, default='<sos>')
args = parser.parse_args()

with open(args.sent_file, encoding='utf-8') as fs, open(args.parsed_file, encoding='utf-8') as fp, open(args.out_parse, 'w', encoding='utf-8') as fout:
    counter = 0
    for in_sline, in_pline in zip(fs, fp):
        counter += 1
        if(counter % 1000 == 0):
            sys.stdout.write(f"\rLine {counter} done!")
            sys.stdout.flush()
        sline = in_sline.strip()
        pline = in_pline.strip()
        # Replace all the instances of REAL words plus ) with )
        sline_split = sline.split()
        
        for wd in sline_split:
            wd_replace = ' '+wd+')'            
            pline = pline.replace(wd_replace, ' <unit>)')            
        
        stack = []
        out_line = []
        # print(sline)
        # print(pline)
        pline_split = pline.split()        
        # Special start token - "<sos>"
        for wd in pline_split:
            if(wd == '('):
                stack.append(args.start_token)
                out_line.append('/'+args.start_token)
            # elif(wd == '(QUOTE'):

            elif(wd[0] == '('):
                stack.append(wd[1:])
                out_line.append('/'+wd[1:])
            # Guaranteed to be all right-parens
            elif wd.startswith('<unit>'):
                # This tag was prepended with a start slash. Remove the slash.
                t = stack.pop()
                out_line.pop()
                out_line.append(t)
                wd = wd.replace('<unit>)', '')
                for x in wd:
                    t = stack.pop()
                    out_line.append(t+'/') # End slash

            elif(wd[0] == ')'):
                for x in wd:
                    try:                    
                        t = stack.pop()
                    except:
                        print(counter)
                        print(sline)
                        print(out_line)
                        print(wd)
                        exit()
                    out_line.append(t+'/') # End slash                    
            else:
                print(sline)
                print(out_line)
                print(wd)
                print(counter)
                assert(False)
        
        fout.write(" ".join(out_line))
        fout.write("\n")
    print("")
        

        # Now replace all right brackets with corresponding "END" tokens
        
# with open(args.sent_file, encoding='utf-8') as fs, open(args.parsed_file, encoding='utf-8') as fp, open(args.out_parse, 'w', encoding='utf-8') as fout:
#     counter = 0
#     for in_sline, in_pline in zip(fs, fp):
#         counter += 1
#         if(counter % 1000 == 0):
#             sys.stdout.write(f"\rLine {counter} done!")
#             sys.stdout.flush()
#         sline = in_sline.strip()
#         pline = in_pline.strip()
#         # Replace all the instances of REAL words plus ) with )
#         sline_split = sline.split()
        
#         for wd in sline_split:
#             wd_replace = ' '+wd+')'
#             if(wd == '"'):
#                 wd_replace = ' ``)'
#             pline = pline.replace(wd_replace, ' <unit>)')
#             if(wd == '"'):
#                 pline = pline.replace(" '')", ' <unit>)')
        
#         stack = []
#         out_line = []
#         print(sline)
#         print(pline)
#         pline_split = pline.split()        
#         # Special start token - "<sos>"
#         for wd in pline_split:
#             if(wd == '('):
#                 stack.append(args.start_token)
#                 out_line.append('/'+args.start_token)
#             # elif(wd == '(QUOTE'):

#             elif(wd[0] == '('):
#                 stack.append(wd[1:])
#                 out_line.append('/'+wd[1:])
#             # Guaranteed to be all right-parens
#             elif wd.startswith('<unit>'):
#                 # This tag was prepended with a start slash. Remove the slash.
#                 t = stack.pop()
#                 out_line.pop()
#                 out_line.append(t)
#                 wd = wd.replace('<unit>)', '')
#                 for x in wd:
#                     t = stack.pop()
#                     out_line.append(t+'/') # End slash

#             elif(wd[0] == ')'):
#                 for x in wd:
#                     t = stack.pop()
#                     out_line.append(t+'/') # End slash                    
#             else:
#                 print(out_line)
#                 print(wd)
#                 print(counter)
#                 assert(False)

#         print("")
#         fout.write(" ".join(out_line))
        

#         # Now replace all right brackets with corresponding "END" tokens
        
