def df_add_scorecol(data,
                    metric,
                    win_threshold=.66,
                    lose_threshold=.33,
                    win_points=2,
                    lose_points=-2,
                    tie_points=0,
                    score_col_name='score'):

    '''Adds an arbitrary score based on a metric column
    and parameters that set the score boundaries. Takes
    the win and lose thresholds, and applies them to
    inter quantile range on the metric column. What is above
    win threshold becomes a win, and what is below loss becomes
    a lose, and everything in between becomes a tie.

    data | DataFrame | a pandas dataframe with the data
    metric | str | the column to be used for basis of scoring
    win_threshold | float | the iqr at which point a win is consider
    lose_threshold | float | the iqr at which point a lose is considered
    win_points | int | the number of points for a win
    lose_points | int | the number of points for a lose
    tie_points | int | the number of points for a tie (between win and lose)
    score_col_name | str | name to be used for the score column

    '''

    win = data.val_f1score.quantile(win_threshold)
    lose = data.val_f1score.quantile(lose_threshold)

    result = []

    for i in data[metric].values:

        # wins the game
        if i >= win:
            result.append(win_points)

        # loses the game
        elif i <= lose:
            result.append(lose_points)

        # game is tied
        else:
            result.append(tie_points)

    data[score_col_name] = result

    return data
