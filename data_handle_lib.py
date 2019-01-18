import numpy as np
import pandas as pd


def read_raw_data(data_path, metadata_path):
    df_metadata = pd.read_csv(metadata_path)
    df_data = pd.read_csv(data_path, sep='\t')

    metadata_descr = df_metadata.iloc[0]  # first row is a description of each column
    df_metadata = df_metadata.drop([0], axis=0)

    return [df_data, df_metadata, metadata_descr]


def merge_data(df_raw_data, df__raw_metadata, tax_rank='genus',
               metadata_useful=['sample_name', 'collection_date', 'common_sample_site', 'host_individual',
                                'mislabeled'],
               drop_unclassified=True, drop_mislabeled=True, index_time=True):
    '''
    Function that takes in pandas data frames of data and metadata, cleans them up and returns a merged version of the two,
    indexed over the time since the first sampling (in days)

    metadata_useful: array of strings that refer to the column names in the metadata to be kept for the returned dataframe.
    '''
    if (metadata_useful):
        # Only keep useful metadata columns
        df__raw_metadata = df__raw_metadata.loc[:, metadata_useful]

    # Index on sample_name so we can combine the meta_data with the data
        df__raw_metadata = df__raw_metadata.set_index('sample_name')
    # Change file names to match those in meta_data
        df_raw_data.columns = [col_title.split('.', 1)[0] for col_title in df_raw_data.columns]
    # Drop unclassified samples
    if (drop_unclassified):
        df_raw_data = df_raw_data.drop(index=(df_raw_data[df_raw_data[tax_rank].str.contains("unclassified")]).index)

    # Make the metadata and the data dataframes match and...
    df_data_1 = pd.concat([(df_raw_data.T).iloc[0:6], (df_raw_data.T).iloc[6:].sort_index()])
    df_metadata_1 = pd.concat(
        [pd.DataFrame(np.nan, index=df_data_1.iloc[0:6].index.values, columns=df__raw_metadata.columns), df__raw_metadata])
    # ...concatanate them
    df_data = pd.concat([df_metadata_1, df_data_1], axis=1, sort=False)

    # Index the dataframe on time since sampling (in days)
    if (index_time):
        df_data.insert(0, 'sample_name', df_data.index.values)  # to store once the index will be reset
        # Set index to days (in float) since first sampling
        df_data['collection_date'] = pd.to_datetime(df_data['collection_date'])
        df_data.loc[df_data['collection_date'].first_valid_index():] = \
            (df_data.loc[df_data['collection_date'].first_valid_index():].sort_values(by='collection_date')).values
        t0 = df_data['collection_date'][6]
        df_data['days'] = (df_data['collection_date'] - t0).astype(
            'timedelta64[D]')  # Time difference in days (converted to floats)
        df_data = df_data.set_index('days')

    # Eliminate mislabeled data
    if (drop_mislabeled):
        df_data = df_data[(df_data.mislabeled == 'n') | (df_data['mislabeled'].isnull())]

    return df_data

def df_normalize(df, along_row = True):
    '''
    :param
        df: pandas data frame
        along_row: boolean
            If True, the normalization is done along the row. Otherwise, along the column
    :return:
        pandas data frame
            normalized version of df
    '''
    # The replace(0, 1) is needed to avoid division by zero in case all elements of the row/column are zero
    if(along_row):
        return df.div(df.sum(axis=1).replace(0, 1), axis=0)
    else:
        return df.div(df.sum(axis=0).replace(0, 1), axis=1)