import numpy as np

class Levenshtein():

    @staticmethod
    def claculate_distance(old_word, new_word):
        rows = len(old_word) + 1
        cols = len(new_word) + 1
        dist = np.zeros((rows, cols), dtype=np.int)
        for i in range(1, rows):
            dist[i][0] = i
        
        for j in range(1, cols):
            dist[0][j] = j

        for row in range(1, rows):
            for col in range(1, cols):
                if old_word[row - 1] == new_word[col - 1]:
                    cost=0
                else:
                    cost = 2
                dist[row][col] = min(dist[row-1][col] + 1,
                                     dist[row][col-1] + 1,
                                     dist[row-1][col-1] + cost)
        return dist[rows-1,cols-1]       
        