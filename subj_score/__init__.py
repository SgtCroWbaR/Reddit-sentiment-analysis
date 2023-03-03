import pandas as pd


# TODO: rows[0:1] promeniti u ceo red samo ako imam saznanje
#       uloge koju rec ima unutar recenice

class Score:

    def __init__(self, paragraph: str, dictionary: pd.DataFrame):
        self.words = paragraph.strip().split(' ')
        self.dictionary = dictionary
        self.score_array = []
        self.rel_score = 0
        self.__calc_score()

    def __calc_score(self):
        for word in self.words:
            rows = self.dictionary.query('@word in word').to_dict(orient='records')

            if len(rows) == 0:
                self.score_array.append(0)

            for row in rows[0:1]:
                if (row['word'].startswith(word) and row['stemmed'] == 'y') or row['word'] == word:
                    self.score_array.append(row['single_score'])
        self.rel_score = sum(self.score_array) / len(self.score_array)

    def __str__(self):
        res = ""
        for i in range(len(self.words)):
            res += str(self.words[i]) + "(" + str(self.score_array[i]) + ")" + " "
        res: str = res.strip()
        return res
