from torch.utils.data import WeightedRandomSampler

def get_sampler(dataset):
    srcs = dataset.sources
    sources_count = [list(srcs.values()).count(source) for source in set(list(srcs.values()))]
    sources_count_dict = dict(zip(list(set(list(srcs.values()))), sources_count))
    weights = [1/sources_count_dict[srcs[image_id]] for image_id in list(srcs.keys())]
    sampler = WeightedRandomSampler(weights, len(dataset))
    return sampler