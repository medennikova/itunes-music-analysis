def read_db_in_pandas(db):
    """Read the DB and return a DF.
    """
    import pandas as pd
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

