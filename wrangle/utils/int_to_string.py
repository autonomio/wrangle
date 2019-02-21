def int_to_chars(x):

    padding = len(str(max(x)))

    out = [list(str(i).zfill(padding)) for i in x]

    return out
