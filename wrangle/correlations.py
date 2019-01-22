from .convert_to_numeric import convert_to_numeric


def corr_pearson(data):
    return convert_to_numeric(data).corr()

def corr_spearman(data):
    return convert_to_numeric(data).corr('spearman')

def corr_kendall(data):
    return convert_to_numeric(data).corr('kendall')