# ''' the code for ML ''''
import numpy as np
import pandas as pd
from sklearn import preprocessing

#////////////////////////////////////////////////////////////////
def main(file_name):
    data = pd.read_csv(file_name, sep=',')
    labels = ['cut', 'color','clarity']

    for n in range(len(labels)):
        label_encoder = preprocessing.LabelEncoder()
        tmp = data[labels[n]]
        label_encoder.fit(tmp)
        data[labels[n]] = label_encoder.transform(tmp)

    data.to_csv('modified.csv',sep=",")
    return data

#////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    file_name = 'diamonds.csv'
    main(file_name)