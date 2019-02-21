from .df_to_numeric import df_to_numeric


def df_corr_any(data, correlation):
    return df_to_numeric(data).corr(correlation)
