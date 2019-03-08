# ''' the code for ML ''''
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet, LassoCV
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier

#////////////////////////////////////////////////////////////////
def main(file_name):
    data = pd.read_csv(file_name, sep=',')
    labels = ['cut', 'color','clarity']

    for n in range(len(labels)):
        label_encoder = preprocessing.LabelEncoder()
        tmp = data[labels[n]]
        label_encoder.fit(tmp)
        data[labels[n]] = label_encoder.transform(tmp)

    data.to_csv('modified.csv', sep=",")
    data = data.drop(columns=['No','cut','color','clarity'])


    label_data = data['price']
    X_Data = data.drop(columns=['price'])
    train_X, test_X, train_Y, test_Y = train_test_split(X_Data, label_data, test_size=0.30, random_state=42)

    clf_MLP = LinearRegression()
    # clf_RFC = DT()
    # set to 10 folds
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X_Data, label_data):
        # specific ".loc" syntax for working with dataframes
        x_train, x_test = X_Data.loc[train_index], X_Data.loc[test_index]
        y_train, y_test = label_data[train_index], label_data[test_index]
        # clf_MLP.fit(x_train, y_train)

        clf_MLP.fit(x_train, y_train)

    y_pred_RFC = clf_MLP.predict(X_Data)
    scores_RFC = clf_MLP.score(X_Data, label_data)

    # y_pred_MLP = clf_MLP.predict(X_Data)
    # scores_MLP = clf_MLP.score(X_Data, label_data)
    print("Accuracy is ", np.round(scores_RFC * 10000) / 100, "%")

    return_data = pd.DataFrame([], columns=['Price', 'est_price'])
    return_data['Price']=label_data
    return_data['est_price']=y_pred_RFC
    return_data.to_csv('estimatedResult.csv', sep=',')

    return data

#////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    file_name = 'diamonds.csv'
    main(file_name)