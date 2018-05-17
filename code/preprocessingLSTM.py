# -*- coding: utf-8 -*-
import numpy as np

class proLSTM(object):

    def lookup_pretrained(self,vectorpath,tokenizer,wordvector_dim):
        word_dict=tokenizer.word_index
        wordlist=list(word_dict.keys())
        lookup_table=np.zeros((len(wordlist),wordvector_dim))
        f = open(vectorpath,encoding='UTF8')
        for line in f:
            values = line.split(' ')
            word = values[0]
            if word in wordlist:
                vectors = np.asarray(values[1:], dtype='float32')
                lookup_table[word_dict[word]-1,:] = vectors
        f.close()
        return lookup_table

    def label_line(self,df_line):
        if df_line['polarity']=='1':
            return [1, 0, 0]
        elif df_line['polarity']=='-1':
            return [0,1,0]
        else:
            return [0,0,1]

    def getlabel(self,cleandata):
        labels=cleandata.apply(self.label_line,axis=1)
        return labels

    def replace_wtag(self,df_line,tokenizer,defauttag):
        message=df_line['message']
        message_lst=message.split(' ')
        targets=df_line['target']
        target_lst = targets.split(' ')
        for word in message_lst:
            if not word in tokenizer.word_index:
                message=message.replace(word,defauttag)
        for word in target_lst:
            if not word in tokenizer.word_index:
                targets=targets.replace(word,defauttag)

        df_line['message']=message
        df_line['target']=targets

        return df_line

    def label_str(self,labelvector):
        largestindex=np.argmax(labelvector)
        if largestindex==0:
            return '1'
        elif largestindex==1:
            return '-1'
        else:
            return '0'
