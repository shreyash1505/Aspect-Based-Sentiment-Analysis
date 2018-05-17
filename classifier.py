# -*- coding: utf-8 -*-

from cleaner import Cleaner
from preprocessingLSTM import proLSTM
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk import pos_tag, word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import optimizers
import pickle

#path for the pretrained vectors
GLOVE_DATASET_PATH = '../data/glove.twitter.27B.200d.txt'
WORDVECTOR_DIM=200
DIST_TARGET=10
POS_DIM=36
DIM_WORD=WORDVECTOR_DIM+DIST_TARGET+POS_DIM
#dict for coding the POSTag
POSDICT={'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,'LS':9,'MD':10,'NN':11,'NNS':12,
         'NNP': 13,'NNPS':14,'PDT':15,'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,'SYM':23,'TO':24,'UH':25,
         'VB': 26,'VBD':27,'VBG':28,'VBN':29,'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34,'WRB':35}


class Classifier:
    """The Classifier"""
    def __init__(self):
        self.tokenizer = Tokenizer(filters='')
        self.defauttag='<hashtag>'
        self.maxlen_sent=100

    def clean(self,tfile):
        
        # load data
        data = pd.read_csv(tfile, sep=',', header=None,
                           names=['text_id','message','target', 'startend','polarity' ],dtype=str)
        # clean the data
        print(data.dtypes)
        cleaner = Cleaner()
        new_data = cleaner.remove_punctuation_dataframe(data)
        new_data = cleaner.remove_digits_dataframe(new_data)
        new_data = cleaner.lemmatization_dataframe(new_data)
        new_data = cleaner.lower_case(new_data)
        return new_data

    def inputlstm_line(self,df_line):
        POSlst=pos_tag(word_tokenize(df_line['message']))
        seq_line=pad_sequences(self.tokenizer.texts_to_sequences([df_line['message']]), maxlen=self.maxlen_sent)
        tpmatrix=np.zeros((self.maxlen_sent,DIM_WORD))

        tpmatrix[:,WORDVECTOR_DIM]=1

        targets=df_line['target'].split()
        index_dict=self.tokenizer.word_index
        indexlst=[index_dict.get(word,None) for word in targets]
        tplst=[]
        for i in indexlst:
            tplst+=np.where(seq_line==i)[1].tolist()
        midpos=int(DIST_TARGET/2)
        tpmatrix[tplst,WORDVECTOR_DIM + midpos]=1
        if tplst[0] - 1 >= 0:
            tpmatrix[tplst[0]-1, WORDVECTOR_DIM + midpos-1] = 0.5
        if tplst[0] - 2 >= 0:
            tpmatrix[tplst[0]-2, WORDVECTOR_DIM + midpos-2] = 0.2
        if tplst[0] - 3 >= 0:
            tpmatrix[tplst[0]-3, WORDVECTOR_DIM + midpos-3] = 0.1

        if tplst[-1] + 1 <= self.maxlen_sent-1:
            tpmatrix[tplst[-1] + 1, WORDVECTOR_DIM + midpos+1] = 0.5
        if tplst[-1] + 2 <= self.maxlen_sent-1:
            tpmatrix[tplst[-1] + 2, WORDVECTOR_DIM + midpos+2] = 0.2
        if tplst[-1] + 3 <= self.maxlen_sent-1:
            tpmatrix[tplst[-1] + 3, WORDVECTOR_DIM + midpos+3] = 0.1


        for i in range(1,self.maxlen_sent+1):
            tp=seq_line[0,-i]
            tpmatrix[-i, 0:WORDVECTOR_DIM] = self.wordvec_lookup[tp-1]
            if tp>0:
                posindex=POSDICT[POSlst[-i][1]]
                tpmatrix[-i, WORDVECTOR_DIM + DIST_TARGET+posindex] = 1
        return tpmatrix.tolist()


    def genInput_LSTM(self,cleandata):
        input_matrix=cleandata.apply(self.inputlstm_line,axis=1)
        return input_matrix.as_matrix()


    def train(self, trainfile):
        processor=proLSTM()
        traindata = self.clean(trainfile)
        train_addtag=traindata.copy()
        train_addtag['message'][0]+=' '+self.defauttag
        self.tokenizer.fit_on_texts(train_addtag['message'])
        self.wordvec_lookup=processor.lookup_pretrained(GLOVE_DATASET_PATH,self.tokenizer,wordvector_dim=WORDVECTOR_DIM)
        input_train = (self.genInput_LSTM(traindata)).tolist()

        train_labels = np.array(processor.getlabel(traindata).tolist())


        model = Sequential()
        model.add(LSTM(DIM_WORD, input_shape=(self.maxlen_sent, DIM_WORD),dropout=0.3,recurrent_dropout=0.3))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        adam_op = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
        rmsprop = optimizers.RMSprop(lr=0.001, rho=0.99, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=rmsprop,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
        self.history = model.fit(np.array(input_train), np.array(train_labels), validation_split = 0.1, epochs=50,verbose=2)
        self.model=model

    def predict(self, devfile):
        processor = proLSTM()
        devdata = self.clean(devfile)
        devdata=devdata.apply(processor.replace_wtag, axis=1,tokenizer=self.tokenizer,defauttag=self.defauttag)
        input_dev = (self.genInput_LSTM(devdata)).tolist()
        
        pred_test=self.model.predict(np.array(input_dev))
        returnlst=np.apply_along_axis(processor.label_str, 1, pred_test)
        returnlst=returnlst.tolist()
        for i in range(len(returnlst)):
            if returnlst[i]=='-':
                returnlst[i]='-1'
        return returnlst





