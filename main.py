from subj_score import Score
from fuzzy import FuzzyInput, FuzzyOutput, Point
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    comments = pd.read_csv('data/dataset/reddit_comments.csv')
    subj_patterns = pd.read_csv('data/lexicon/subj_patterns.csv')

    X, y = comments['clean_comment'], comments['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y)

    budism = comments.at[0, 'clean_comment']
    budism_score = Score(budism, subj_patterns)

    fuz = FuzzyInput('kratki',[Point(50,0), Point(100,1), Point(400, 1), Point(600,0)], 405)
    print(fuz.mu)

