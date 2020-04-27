import numpy as np
from collections import Counter
import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import csv
import re

def main():
    labels = []
    with open("IMDB_labels.csv", 'r') as file:
        labels = pd.read_csv(file)['sentiment']

    # Importing the dataset
    imdb_data = pd.read_csv('IMDB.csv', delimiter=',')

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=4000,
        min_df=30
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    pos_text = []
    neg_text = []
    pos_count = 0
    neg_count = 0
    for i, review in enumerate(imdb_data['review'][0:30000]):
        if (labels[i] == "positive"):
            for word in list(set(vocabulary) & set(clean_text(review).split())):
                pos_text.append(word)
            pos_count += 1
        else:
            for word in list(set(vocabulary) & set(clean_text(review).split())):
                neg_text.append(word)
            neg_count += 1

    pos_dict = Counter(pos_text)
    neg_dict = Counter(neg_text)
    prob_pos = pos_count/(pos_count+neg_count)
    prob_neg = neg_count/(pos_count+neg_count)


    correct = naive_bayes(imdb_data['review'][30000:40000], pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, labels, vocabulary, 1, 30000)
    print("Testing Accuracy: ", correct/10000)

    # predictions = naive_bayes_predict(imdb_data['review'][40000:50000], pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, vocabulary, 1.0)
    # with open("test-predictions3.csv", 'w') as file:
    #     for p in predictions:
    #         file.write(str(p)+"\n")

def naive_bayes(reviews, pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, labels, vocabulary, alpha, offset):
    correct = 0
    for i, review in enumerate(reviews):
        conditional_pos = 0.
        conditional_neg = 0.
        for word in clean_text(review).split():

            conditional_pos += math.log(((pos_dict[word]+alpha)/(len(pos_text)+len(vocabulary)*alpha)) * prob_pos)
            conditional_neg += math.log(((neg_dict[word]+alpha)/(len(neg_text)+len(vocabulary)*alpha)) * prob_neg)

        if ((conditional_pos > conditional_neg and labels[i+offset] == "positive") or (conditional_pos < conditional_neg and labels[i+offset] == "negative")):
            correct += 1

    return correct

def naive_bayes_predict(reviews, pos_dict, neg_dict, pos_text, neg_text, prob_pos, prob_neg, vocabulary, alpha):
    predictions = []
    for i, review in enumerate(reviews):
        conditional_pos = 0.
        conditional_neg = 0.
        for word in clean_text(review).split():
            conditional_pos += math.log(((pos_dict[word]+alpha)/(len(pos_text)+len(vocabulary)*alpha)) * prob_pos)
            conditional_neg += math.log(((neg_dict[word]+alpha)/(len(neg_text)+len(vocabulary)*alpha)) * prob_neg)

        if (conditional_pos > conditional_neg):
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions

def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text



if __name__ == '__main__':
    main()
