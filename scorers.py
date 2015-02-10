import numpy as np
import pandas as pd


class Linear(object):

    """
    A 'piecewise linear' scorer.

    Parameters
    ----------

    values : array_like
             The sequence of values.

    scores : array_like
             The sequence of scores (default: [0, 10])

    The length of `values` and `scores` must be the same.
    """

    def __init__(self, values, scores=[0, 10], weight=1):
        self.values = np.array(values)  # This copies the values.
        self.scores = np.array(scores)  # This copies the values.
        self.weight = weight

    def __call__(self, data):
        """
        Score the data according to this score model.
        """
        # This if statement 'block' makes sure `data` is in the right
        # 'format' (object type)
        if data.__class__ is pd.Series:
            data = data.values
        elif np.ndarray not in data.__class__.__mro__:
            if not np.iterable(data):
                data = [data]
            data = np.array(data)
        return np.interp(data, self.values, self.scores)

    def __repr__(self,):
        outstr = '<ScoreLinear>\nWEIGHT: %4.2f\n' % (self.weight, )
        outstr += 'Values: %s\n' % self.values
        outstr += 'Scores: %s\n' % self.scores
        return outstr

    def __copy__(self,):
        return self.__class__(self.values, self.scores, self.weight)


class Table(object):

    """
    A 'look-up table' scorer.

    Parameters
    ----------

    table  : array_like
             The format of the tables is:
                  [minval1, maxval1, score1]
                  [minval2, maxval2, score2]
                  ...
                  [minvalN, maxvalN, scoreN]
             Values in the range minval1 to maxval1 are given score1,
             and so on.

    weight : float
             The weight to give this scorers relative to others.
    """

    def __init__(self, table, weight=1):
        if np.array not in table.__class__.__mro__:
            self.table = np.array(table)
        self.weight = weight

    def __copy__(self,):
        return self.__class__(self.table.copy(), weight=self.weight)

    copy = __copy__

    def __repr__(self,):
        outstr = '<ScoreTable>\nWEIGHT: %4.2f\n------ Range --------- :  Score\n' % (self.weight, )
        for v in self.table:
            outstr += ' %9.1f - %9.1f : %5.1f\n' % tuple(v)
        return outstr

    def __call__(self, data):
        """
        This function uses the score table (or array) `self.table` to calculate
        the scores for the data points in `data`.
        """

        # This if statement 'block' makes sure `data` is in the right
        # 'format' (object type)
        if data.__class__ is pd.Series:
            data = data.values
        elif np.ndarray not in data.__class__.__mro__:
            if not np.iterable(data):
                data = [data]
            data = np.array(data)

        # First initialize the output array.
        # Values not specified in the table should be NaN.
        out = np.zeros(data.shape) * np.NaN

        # Now perform a 'for loop' over the min, max, score values:
        for mn, mx, score in self.table:
            # Test that data is within the range, and assign scores:
            out[((mn < data) & (data <= mx))] = score
        return out  # Return the output data from the function.
