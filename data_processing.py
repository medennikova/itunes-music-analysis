from sklearn import preprocessing
import pandas as pd
import numpy as np

def read_db_in_pandas(db):
    """Read the DB and return a DF.
    """
    # transpose data to have songs as rows
    df = pd.DataFrame(dict(db)).T

    # remove rows with no data for a song
    df_clean = df[df['No_data'] != 1]

    # convert columns into numbers
    df_clean = df_clean.convert_objects(convert_numeric=True)

    # convert index into a column
    df_clean.reset_index(level=0, inplace=True)
    df_clean.rename(columns = {'index': 'song_title'},
                    inplace=True)

    # remove the 'No_data' column
    df_clean.drop('No_data', 1,
                  inplace=True)
    return df_clean


def target_to_numerical_format(df):
    """Encode labels with values between 0 and n_classes-1
    using LabelEncoder scikit-learn function.
    Return an array with target labels in numerical format.
    """
    le = preprocessing.LabelEncoder()

    # list the three categories
    categories = list(pd.unique(df.category.ravel()))
    le.fit(categories)

    # create an array with target labels
    target = le.transform(df['category'])
    return target

def convert_df_to_array(df):
    """Convert DF to a Numpy array so that each track
    forms a row, and features form columns.
    Return an array.
    """
    # columns with numerical values
    cols = df.select_dtypes(exclude=[object]).columns

    # convert df to a numerical (Numpy) array
    data = np.nan_to_num(df[cols].values)

    return data

def standardize_data(train_data, test_data):
    """Perform standardization of features
    using StandardScaler() module.
    Returns train_std, test_std.
    """
    # train the model
    std_scale = preprocessing.StandardScaler().fit(train_data)
    # perform standardization of features in the training set
    train_std = std_scale.transform(train_data)

    # perform standardization of features of test data
    test_std = std_scale.transform(test_data)

    return train_std, test_std

def reduce_dimensions(train_data, test_data):
    """Perform PCA to reduce the dimensionality
    of both sets.
    Returns train_2d, test_2d.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(train_data)
    train_2d = pca.transform(train_data)
    test_2d = pca.transform(test_data)

    return train_2d, test_2d
