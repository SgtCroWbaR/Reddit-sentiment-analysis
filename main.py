from subj_score import Score
from fuzzy import FuzzyInput, FuzzyOutput, Point
from word_embeddings import Embeddings
from attention import MultiHeadAttention
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    # comments = pd.read_csv('data/dataset/reddit_comments.csv')
    # subj_patterns = pd.read_csv('data/lexicon/subj_patterns.csv')
    dictionary = np.load('glove.6B/glove_100d_dict.npy', allow_pickle=True).item()


    # X, y = comments['clean_comment'], comments['category']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y)
    #
    # budism = comments.at[0, 'clean_comment']
    # budism_score = Score(budism, subj_patterns)
    #
    # fuz = FuzzyInput('kratki', [Point(50, 0), Point(100, 1), Point(400, 1), Point(600, 0)], 405)
    # print(fuz.mu)

    # comments_embedings = Embedings(comments, dictionary)
    testing_embeding = Embeddings(['cat dog', 'the rain cloud'], dictionary)

    # print(len(testing_embeding.texts_embedings))
    # print(len(testing_embeding.texts_embedings[0]))
    # print(testing_embeding.texts_embedings)

    # FUTURE PROBLEM:
    # najveca paznja je na samog sebe


    # testing_atention = MultiHeadAttention(testing_embeding.texts_embedings, 1)
    # # print(testing_atention.scores)
    # # print(testing_atention.normalised_scores)
    # print(testing_atention.reweighted)

    testing_atention2 = MultiHeadAttention(testing_embeding.texts_embedings, 2)
    print(testing_atention2.reweighted)



    # TODO:
    # treniranje LSTM modela

    # TODO:
    # RNN

    # TODO:
    # fuzzy pravila nad treniranim podacima
