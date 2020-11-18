import numpy as np
import pandas as pd


user_emb_matrix = np.array(pd.read_csv(r'util/user_emb.csv', header=None))
# print(user_emb_matrix)
a = [1, 5]
print(user_emb_matrix[a])