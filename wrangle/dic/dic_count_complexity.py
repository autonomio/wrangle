def dic_count_complexity(dic):

    out = 1
    for key in dic.keys():
        out *= len(dic[key])

    return out
