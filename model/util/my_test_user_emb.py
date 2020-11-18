import pandas as pd
import numpy as np

data = pd.read_csv(r'../data/train_data.csv', usecols=['userId', 'gerne'])
#    print(data)

data['tmp'] = data['gerne'].str.split('[', expand=True)[1]
data['tmp1'] = data['tmp'].str.split(']', expand=True)[0]

#    print(data)
user = np.array(data['userId'])
attr = np.array(data['tmp1'])

print(len(user))
print(len(attr))

user_present = np.zeros(shape=(6040, 18), dtype=np.int32)

for i in range(len(user)):
    attr_list = np.int32(attr[i].split(','))
    for j in attr_list:
        user_present[user[i]][j] += 1.0

save = pd.DataFrame(user_present)
# print(save)
save['Col_sum'] = save.apply(lambda x: x.sum(), axis=1)
save = np.array(save, dtype=np.float32)
print(save)
for i in range(6040):
    tt = save[i][-1]

    if tt != 0.0:
        for j in range(18):
            save[i][j] = save[i][j] / tt

save = pd.DataFrame(save)
print(save)