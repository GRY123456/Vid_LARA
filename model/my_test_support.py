import numpy as np
import pandas as pd

train_data = np.array(pd.read_csv(r'data/train_data.csv', usecols=['userId', 'itemId', 'gerne']))

test_data = pd.read_csv(r'data/test_data.csv', usecols=['itemId', 'gerne'])
print(test_data)
print(len(test_data))
test_data.drop_duplicates(keep='first', inplace=True)
print(test_data)
print(len(test_data))
