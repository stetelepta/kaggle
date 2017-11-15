import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
from matplotlib import gridspec # for subplots
from utils.data_utils import split_column, convert_categories
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle


def process_titanic_data():
    # read csv
    train_df = pd.read_csv('../input/train.csv', delimiter=',', quotechar='"')
    test_df = pd.read_csv('../input/test.csv')

    # Split "Name" into "LastName", "Title" and "FirstNames"
    train_df = split_column(train_df, 'Name', ['LastName', 'Title', 'FirstNames'], [', ', '\. '])
    test_df = split_column(test_df, 'Name', ['LastName', 'Title', 'FirstNames'], [', ', '\. '])

    # view top 5 rows
    train_df.head()

    train_df = convert_categories(train_df)
    test_df = convert_categories(test_df)

    features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']

    train_df = train_df[features].copy()
    test_df = test_df[features[1:]].copy()

    # drop records with Nan
    train_df = train_df.dropna()
    # test_df = test_df.dropna()

    # convert data to (m, n_x) numpy array
    x_np_train = train_df.as_matrix()
    x_test = test_df.as_matrix()

    # 1. split data into two parts: 0.8 (train), 0.2 (validation)
    x_train, x_val = train_test_split(x_np_train, test_size=0.2, shuffle=True)

    # create binary labels
    lb = preprocessing.LabelBinarizer()

    y_train = lb.fit_transform(x_train[:, 0])
    y_val = lb.fit_transform(x_val[:, 0])

    # strip gait_type from input
    x_train = x_train[:, 1:]
    x_val = x_val[:, 1:]

    # use validation set for calculating mean and std
    x_measure = x_train

    # calculate mean and st deviation
    x_mean = np.mean(x_measure, axis=0)
    x_std = np.std(x_measure, axis=0)

    # normalize data so we get a distribution with mean=0 and std=1
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    # save measured mean and std as pickle
    pickle.dump({'x_mean': x_mean, 'x_std': x_std, 'classes': lb.classes_, 'features': features[1:]}, open("meta.pkl", "wb"))

    print("x_train.shape:", x_train.shape)
    print("x_val.shape:", x_val.shape)
    print("x_test.shape:", x_test.shape)    

    # dump variables
    pickle.dump({
        'x_train': x_train, 
        'x_val': x_val,
        'x_test': x_test,
        'y_train': y_train,
        'y_val': y_val
    }, open("processed_data.pkl", "wb"))

if __name__ == "__main__":
    process_titanic_data()
