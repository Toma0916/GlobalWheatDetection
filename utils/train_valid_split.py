
def debug_split(dataframe):
    image_ids = dataframe['image_id'].unique()
    train_data_size = 100
    valid_data_size = 20
    train_ids = image_ids[:train_data_size]
    valid_ids = image_ids[train_data_size:(train_data_size+valid_data_size)]
    return train_ids, valid_ids


def random_split(dataframe, train_rate=0.8):
    image_ids = dataframe['image_id'].unique()

    image_num = len(image_ids)
    train_data_size = int(image_num * train_rate)
    valid_data_size = int(image_num - train_data_size)
    train_ids = image_ids[:train_data_size]
    valid_ids = image_ids[train_data_size:(train_data_size+valid_data_size)]
    return train_ids, valid_ids


def source_split(dataframe, valid_sources=['ethz_1']):
    """
    ethz_1       51489
    arvalis_1    45716
    rres_1       20236
    arvalis_3    16665
    usask_1       5807
    arvalis_2     4179
    inrae_1       3701
    """
    
    df = dataframe[['image_id', 'source']].drop_duplicates()
    imageid_source_dict = dict(zip(df.image_id, df.source))
    train_ids = [k for k, v in imageid_source_dict.items() if not v in valid_sources]
    valid_ids = [k for k, v in imageid_source_dict.items() if v in valid_sources]
    return train_ids, valid_ids


def train_valid_split(dataframe, config):

    if config['debug']:
        return debug_split(dataframe)

    if 'train_valid_split' not in config['general'].keys():
        config['general']['train_valid_split'] = {'name': 'random', 'config': {'train_rate': 0.8}}
    split_config = config['general']['train_valid_split']
    split_method_list = {
        'debug_split': debug_split,
        'random': random_split,
        'source': source_split
    }

    return split_method_list[split_config['name']](dataframe,  **split_config['config'])