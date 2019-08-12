def create_time_sequence(periods,
                         start_year,
                         start_month,
                         start_date=1,
                         time_format='%m-%Y',
                         period='month'):

    '''Creates a time sequence and outputs a list with the string
    timestamps.

    periods | int | Number of periods/timestamps in the output
    start_year | int | The year to start the periods on
    start_month | int | The month to start the periods on
    start_date | int | The day to start the periods on
    time_format | str | A python datetime string format
    period | str | 'month', 'week', or 'day'

    '''

    import datetime
    from dateutil.relativedelta import relativedelta

    dt = datetime.datetime(start_year, start_month, 1)

    result = []

    for i in range(periods):

        result.append(dt.strftime(time_format))

        if period == 'month':
            dt += relativedelta(months=+1)
        elif period == 'week':
            dt += relativedelta(weeks=+1)
        elif period == 'day':
            dt += relativedelta(days=+1)

    return result
