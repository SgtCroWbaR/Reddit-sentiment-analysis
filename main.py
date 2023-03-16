from subj_score import Score
from fuzzy import FuzzyInput, FuzzyOutput, Point, Rule, LogicOp, defuzzyfy
from word_embeddings import Embeddings
from attention import MultiHeadAttention
from model import Dense, Relu, Linear, Sigmoid, Sequential
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
random_state = 69


if __name__ == "__main__":
    num_of_heads = 10
    comments_scores = pd.read_csv('data/dataset/reddit_comments.csv')

    # TODO: download dataset
    # url = 'https://drive.google.com/file/d/13flpeoEZ9EcY-1sod6yqZjSTJWSm6g8-/view?usp=share_link'
    # output_path = 'glove.6B/glove_100d_dict.npy'
    # gdown.download(url, output_path, quiet=False, fuzzy=True)

    dictionary = np.load('glove.6B/glove_100d_dict.npy', allow_pickle=True).item()
    subj_patterns = pd.read_csv('data/lexicon/subj_patterns.csv')

    with open('data/models/model_full_4L_3epochs_5heads.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Loaded")


    rand_index = 80
    text, ocena = comments_scores['clean_comment'][rand_index], comments_scores['category'][rand_index]

    text_len = len(text)
    length = {'short': FuzzyInput([Point(0, 1), Point(50, 1), Point(100, 0)], text_len),
              'medium': FuzzyInput([Point(50, 0), Point(100, 1), Point(400, 1), Point(600, 0)], text_len),
              'long': FuzzyInput([Point(400, 0), Point(2000, 1)], text_len)}

    score = Score(text, subj_patterns).rel_score
    subjectivity = {'subjective': FuzzyInput([Point(0, 0), Point(0.2, 1)], score),
                    'objective': FuzzyInput([Point(0, 1), Point(0.2, 0)], score)}

    text_vec = MultiHeadAttention(Embeddings([text], dictionary).texts_embedings, num_of_heads).reweighted[0]
    pred = []
    for arr in text_vec:
        pred.append(np.argmax(model.predict(np.reshape(arr, newshape=(100, 1)))))
    counts = np.bincount(np.array(pred))
    padded = np.pad(counts, (0, 3 - counts.shape[0]), 'constant', constant_values=(0, 0))
    scaled = padded / np.sum(padded)
    masked_pred = scaled * np.array([-1, 0, 1])
    sentiment_score = np.sum(masked_pred)
    sentiment = {'negative': FuzzyInput([Point(-0.5, 1), Point(-0.2, 0)], sentiment_score),
                 'neutral': FuzzyInput([Point(-0.4, 0), Point(-0.2, 1), Point(0.2, 1), Point(0.4, 0)], sentiment_score),
                 'positive': FuzzyInput([Point(0.2, 0), Point(0.5, 1)], sentiment_score)}

    type_of_text = {'useless': FuzzyOutput([Point(0, 0), Point(10, 1), Point(20, 1), Point(30, 0)]),
                    'opinion': FuzzyOutput([Point(30, 0), Point(40, 1), Point(50, 1), Point(60, 0)]),
                    'essay': FuzzyOutput([Point(60, 0), Point(70, 1), Point(80, 1), Point(90, 0)])}

    rules = [
        Rule([length['medium'], subjectivity['subjective'], sentiment['negative']], type_of_text['opinion'], LogicOp.AND),
        Rule([length['medium'], subjectivity['subjective'], sentiment['positive']], type_of_text['opinion'], LogicOp.AND),
        Rule([length['long'], subjectivity['objective'], sentiment['positive']], type_of_text['essay'], LogicOp.AND),
        Rule([length['long'], subjectivity['objective'], sentiment['neutral']], type_of_text['essay'], LogicOp.AND),
        Rule([length['medium'], subjectivity['objective'], sentiment['positive']], type_of_text['essay'], LogicOp.AND),
        Rule([length['medium'], subjectivity['objective'], sentiment['neutral']], type_of_text['essay'], LogicOp.AND)]

    res = defuzzyfy(type_of_text.values())
    print(res)
