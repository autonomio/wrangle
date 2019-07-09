def dic_resample_values(params, n):

    import random

    for param in params.keys():
        try:
            params[param] = random.sample(params[param], k=n)
        except ValueError:
            pass

    return params
