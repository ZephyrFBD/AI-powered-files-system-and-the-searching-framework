from gensim.models import word2vec
from tdt import read_txt_files

import logging
logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('start')#logging

directory = r'.\test'#read txt
out = read_txt_files(directory)#read txt
logging.info('readtxt:')#logging
logging.info(out)#logging
input = list(out.values())# the values of the list
logging.info('input:')#logging
logging.info(input)#logging
raw_sentences = input
sentences=[s.split() for s in raw_sentences]#split sentences
logging.info('splitting:')#logging
logging.info(sentences)#logging