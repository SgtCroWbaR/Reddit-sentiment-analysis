from word_embeddings import Embeddings
from attention import MultiHeadAttention
from model import Dense, Relu, Linear, Sigmoid, Sequential
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
random_state = 69


def flatten(X, y):
    xs, ys = [], []
    for paragraph, score in list(zip(X, y)):
        for word in paragraph:
            xs.append(np.reshape(word, (word.shape[0], 1)))
            ys.append(score)
    return xs, np.array(ys)


def train_test_acc(X_train, X_test, y_train, y_test, model_name):

    file = open('data/models/' + model_name + 'STATS.txt', 'w')

    y_train_pred = []
    for X in tqdm(X_train):
        pred = []
        for arr in X:
            pred.append(np.argmax(model.predict(np.reshape(arr, newshape=(100, 1)))))
        counts = np.bincount(np.array(pred))
        y_train_pred.append(np.argmax(counts))
    y_train_true = list(map(lambda x: np.argmax(x), y_train))

    train_acc = accuracy_score(y_train_true, y_train_pred)
    print(f"train  acc: {train_acc}")
    print(f"train: \n{confusion_matrix(y_train_true, y_train_pred)}")
    file.write(f"train  acc: {train_acc}\n")
    file.write(f"train: \n{confusion_matrix(y_train_true, y_train_pred)}\n")

    y_test_pred = []
    for X in tqdm(X_test):
        pred = []
        for arr in X:
            pred.append(np.argmax(model.predict(np.reshape(arr, newshape=(100, 1)))))
        counts = np.bincount(np.array(pred))
        y_test_pred.append(np.argmax(counts))
    y_test_true = list(map(lambda x: np.argmax(x), y_test))

    test_acc = accuracy_score(y_test_true, y_test_pred)
    print(f"test  acc: {test_acc}")
    print(f"test: \n{confusion_matrix(y_test_true, y_test_pred)}")
    file.write(f"test  acc: {test_acc}\n")
    file.write(f"test: \n{confusion_matrix(y_test_true, y_test_pred)}\n")
    file.close()


if __name__ == "__main__":
    comments_scores = pd.read_csv('data/dataset/reddit_comments.csv')
    comments_scores = comments_scores[:]

    # TODO: download dataset
    # url = 'https://drive.google.com/file/d/13flpeoEZ9EcY-1sod6yqZjSTJWSm6g8-/view?usp=share_link'
    # output_path = 'glove.6B/glove_100d_dict.npy'
    # gdown.download(url, output_path, quiet=False, fuzzy=True)

    dictionary = np.load('glove.6B/glove_100d_dict.npy', allow_pickle=True).item()
    subj_patterns = pd.read_csv('data/lexicon/subj_patterns.csv')

    X, y = comments_scores['clean_comment'], comments_scores['category']
    _df_X_train, _df_X_test, _df_y_train, _df_y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    num_of_heads = 15
    X_train = MultiHeadAttention(Embeddings(_df_X_train.values, dictionary).texts_embedings, num_of_heads).reweighted
    X_test = MultiHeadAttention(Embeddings(_df_X_test.values, dictionary).texts_embedings, num_of_heads).reweighted

    y_train = to_categorical(_df_y_train.values, num_classes=3)
    y_test = to_categorical(_df_y_test.values, num_classes=3)



    X_train_flat, y_train_flat = flatten(X_train, y_train)
    # X_test, y_test = flatten(X_test, y_test)


    directory = 'data/models/'

    # model_name = 'model_10k_4L_1epoch.pkl'
    # with open(directory + model_name, 'rb') as file:
    #     model = pickle.load(file)
    # print("Loaded: " + model_name)
    # train_test_acc(X_train, X_test, y_train, y_test, model_name)


    input_size = X_train_flat[0].shape[0]
    layers = [Dense(input_size, 200, Relu()),
              Dense(200, 200, Relu()),
              Dense(200, 50, Relu()),
              Dense(50, 3, Sigmoid()),
              ]
    model = Sequential(layers)


    num_of_epochs = 10
    for i in range(num_of_epochs):
        model.train(X_train_flat, y_train_flat, 1, 0.001)
        model_name = f'model_full_{len(layers)}L_{i+1}epochs_{num_of_heads}heads.pkl'
        with open(directory + model_name, 'wb') as file:
            pickle.dump(model, file)
        print("saved")
        train_test_acc(X_train, X_test, y_train, y_test, model_name)
