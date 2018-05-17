# -*- coding: utf-8 -*-
import string
from nltk.stem import WordNetLemmatizer


class Cleaner(object):
    def lower_case(self, data):
        new_data = data.copy()
        new_data['target'] = new_data['target'].apply(lambda x: x.lower())
        new_data['message'] = new_data['message'].apply(lambda x: x.lower())

        return new_data

    def remove_punctuation_dataframe(self, data):
        new_data = data.copy()
        new_data['message'] = new_data['message'].apply(
            lambda x: self.remove_punctuation(x))
        new_data['target'] = new_data['target'].apply(
            lambda x: self.remove_punctuation(x))

        return new_data

    def remove_punctuation(self, data):
        str_lang = string.punctuation
        for punctuation in str_lang:
            data = data.replace(punctuation, ' ')
        new_data = ' '.join(data.split())
        return new_data

    def remove_digits_dataframe(self, data):
        new_data = data.copy()
        new_data['message'] = new_data['message'].apply(
            lambda x: self.remove_digits(x))
        new_data['target'] = new_data['target'].apply(
            lambda x: self.remove_digits(x))

        return new_data

    def remove_digits(self, data):
        str_lang = string.digits
        for punctuation in str_lang:
            data = data.replace(punctuation, ' ')
        new_data = ' '.join(data.split())
        return new_data

    def lemmatization_dataframe(self, data):
        new_data = data.copy()
        new_data['message'] = new_data['message'].apply(
            lambda x: self.lemmatization(x))

        new_data['target'] = new_data['target'].apply(
            lambda x: self.lemmatization(x))

        return new_data

    def lemmatization(self, data):
        wordnet_lemmatizer = WordNetLemmatizer()
        words = data.split()
        new_data = []

        for word in words:
            word = wordnet_lemmatizer.lemmatize(word, 'v')
            new_data.append(word)

        new_data = ' '.join(new_data)

        return new_data



