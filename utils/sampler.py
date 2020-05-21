from torch.utils.data import WeightedRandomSampler



def balance_sources_sampler(dataset, strength):
    
    srcs = dataset.sources
    sources_count = [list(srcs.values()).count(source) for source in set(list(srcs.values()))]
    sources_count_dict = dict(zip(list(set(list(srcs.values()))), sources_count))

    weights = [1/(sources_count_dict[srcs[image_id]] + 1.0 / strength) for image_id in list(srcs.keys())]
    sampler = WeightedRandomSampler(weights, len(dataset))

    return sampler


def get_sampler(dataset, config):
    if not 'sample' in config.keys():
        config['sample'] = {'name': ''}
    sampler_list = {
        'balance_sources': balance_sources_sampler
    }

    if 'name' not in sampler_list.keys():
        return None
    
    sampler = sampler_list[config['name']](dataset, **config['config'])    
    return sampler