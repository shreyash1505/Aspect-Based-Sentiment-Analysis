# Aspect Based Sentiment Analysis
#### specify the correct training and development file names in tester.py

# Description of the project
## Introduction 
The goal of the project is to design a classifier to predict aspect-based polarities of opinions in
sentences. The polarity labels are positive, negative and neutral. 

## Data set
Each line contains 4 tab-separated fields: the polarity of the opinion, a specific target term, the character offsets of the term (start:end), and the
sentence in which that opinion is expressed.  

## Pretrained word vectors
Download pretrained word vectors from: https://nlp.stanford.edu/projects/glove/
Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)
Update the dimensions appropriately in classify.py according to the GLOVE embedding dimension file chosen. 


## Cleaning Data
Involved removing puncutaions, digits, tokenization, Lemmatization and POS_tagging

## Construction of the input to the LSTM Model
I will explain the process with an example:
"negative SERVICE#GENERAL waitress 20:28 The hostess and the waitress were incredibly rude and did everything they could to rush us out."  

After cleaning the data, we get "the hostess and the waitress were incredibly rude and did everything they could to rush us out" , aspect term "waitress".  

For this message, we construct a 2D matrix: each row of the matrix represents the information of each word in the meassage. The last word in the meassage is "out", so the last row of the 2D matrix should be:   
[ the 50d word vector which represents "out",   
the distance code which represents the distance between "out" and aspect term "waitress",  
POSTag code which represents the POSTag of "out" in this message ]

Here, 100 is the maximum length of words that I choose to represent a message;  
50 is the dimension of pretrained word vector that we use;  
7 is the dimension of the vector for representing the distance of this word to the target word;  
36 is the dimension of the vector for representing the POSTag for this word;


## LSTM model
The layers include
  - LSTM Layer : 100 cells; Each of shape (50+10+36)
  - Dense Layer : 3 cells
  - Activation Layer: "Softmax"
  - Optimizer : "ADAM"
  - Epoches: 25

## Evaluation
  - Accuracy: Achieved an accuracy of 74% on both datasets
  - Macro F1 Scores for laptop dataset and restaurant dataset:
    - Positive class: 78% & 84%
    - Negetive class: 72% & 54%
    - Neutral Class: 57%  & 58%
