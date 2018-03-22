"""
This package contains the scorer classes. Scorers perform scoring of a
single data column of a :class:`.HotSpotCollection` data object. They
are used by incorporated them into one of the scoring models in the
:mod:`.models` module.
"""
import numpy as np
import pandas as pd
from .base import cmaps, maxscore
from matplotlib import pyplot as plt


class Linear(object):

    """
    A 'piecewise linear' scorer.

    Parameters
    ----------
    values : array_like
             The sequence of values.
    scores : array_like
             The sequence of scores (default: [0, 10])

    Notes
    -----

    The length of `values` and `scores` must be the same.

    `values` must either be monotonically increasing or decreasing.
    """

    @property
    def shape(self, ):
        return (len(self), 2)

    def __len__(self, ):
        return len(self.values)

    def to_array(self, col_labels=False):
        dat = np.hstack((self.values[:, None], self.scores[:, None]))
        if not col_labels:
            return dat
        shp = list(self.shape)
        shp[0] += 1
        out = np.empty(shp, dtype='O')
        out[0, :] = 'VALUE', 'SCORE'
        out[1:, :] = dat
        return out

    def __init__(self, values, scores=[0, 10], weight=1):
        self.values = np.array(values)  # This copies the values.
        self.scores = np.array(scores)  # This copies the values.
        self.weight = weight
        if values[0] > values[-1]:
            self.values = self.values[::-1]
            self.scores = self.scores[::-1]

    def __call__(self, data):
        """
        Score the data according to this score model.
        """
        if data.__class__ is pd.Series:
            data = data.values
        return np.interp(data, self.values, self.scores)

    def plot(self, ax=None, vals=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if vals is None:
            minmax = np.array([self.values.min(), self.values.max()])
            minmax[0] = minmax[0] - 0.2 * np.diff(minmax)
            minmax[1] = minmax[1] + 0.2 * np.diff(minmax)
            if self.values.min() > 0 and minmax[0] < 0:
                minmax[0] = 0
            vals = self.space(50, minmax[0], minmax[1])
        ax.plot(vals, self(vals), **kwargs)

    def __repr__(self,):
        outstr = '<ScoreLinear>\nWEIGHT: %4.2f\n' % (self.weight, )
        outstr += 'Values: %s\n' % self.values
        outstr += 'Scores: %s\n' % self.scores
        return outstr

    def __copy__(self,):
        return self.__class__(self.values, self.scores, self.weight)

    def space(self, num, start=None, stop=None):
        """
        Return numbers spaced evenly on this linear scale.

        Parameters
        ----------
        num : int
            The number of samples to generate.
        start : float
            ``self.base ** start`` is the starting value of the
            sequence.
        stop : float
            ``self.base ** stop`` is the final value of the
            sequence.
        """
        if start is None:
            start = self.values.min()
        if stop is None:
            stop = self.values.max()
        return np.linspace(start, stop, num)

    def to_html(self,
                buf,
                legend_vals=None,
                title='',
                cmap=cmaps['score'],
                hline_widths=[1],
                score_col=True,):
        """
        Write the scorer as an html table colored by cmap according to
        the scores.

        Parameters
        ----------
        buf : string or <file buffer>
              The file to write the table to. If it is a string, and
              it does not end in '.htm' or '.html', the latter will be
              appended to the filename.
        legend_vals : array_like
                      An iterable of values (in data units, not score
                      units).
        title : string
                The heading above the data-units column of the table.
        cmap : <matplotlib colormap>
               The colormap to color the table with.
        hline_widths : iterable
                       A list of the line widths to use.
        score_col : bool (default: True)
                    Specify whether to include the 'score' column in
                    the output.
        """
        if legend_vals is None:
            legend_vals = self.space(6)
        dat = pd.DataFrame({title: legend_vals})
        if score_col:
            dat['Score'] = pd.Series(self(dat[title]), dat.index)
        frmt = lambda s: ('%d' % s) if s % 1 == 0 else ('%0.1f' % s)
        tbl = dat.to_html(float_format=frmt, index=False) + '\n'
        tbl = tbl.replace(r'border="1" ', 'style="border-collapse: collapse;"')
        tbl = tbl.replace('<th>',
                          '<th style="border-bottom:solid %0.1fpt;">' % hline_widths[0],
                          1)  # +1 for index
        for irow, val in enumerate(dat.values[:, 0]):
            # This is for the first (index) column:
            if hline_widths is not None:
                lw_txt = ('border-bottom:solid %0.1fpt; ' %
                          hline_widths[(irow + 1) % len(hline_widths)])
            else:
                lw_txt = ''
            tbl = tbl.replace('<th>',
                              '<th style="text-align:center; %s">' % lw_txt,
                              1)
            tbl = tbl.replace('<td>',
                              ('<td style="text-align:center; background-color:rgb(%d, %d, %d);">' %
                               tuple(np.array(cmap(self(val) / maxscore)[:3]) * 255)),
                              1 + score_col)
        if basestring in buf.__class__.__mro__:
            if not (buf.endswith('.htm') | buf.endswith('.html')):
                buf += '.html'
            with open(buf, 'w') as fl:
                fl.write(tbl)
        else:
            buf.write(tbl)


class Log(Linear):

    """
    A 'piecewise logarithmic' scorer.

    Parameters
    ----------
    values : array_like
             The sequence of values in `linear` space.
    scores : array_like
             The sequence of scores (default: [0, 10])

    Notes
    -----

    The length of `values` and `scores` must be the same.

    `values` must either be monotonically increasing or decreasing.

    """

    def space(self, num, start=None, stop=None):
        """
        Return numbers spaced evenly on this log scale.

        Parameters
        ----------
        num : int
            The number of samples to generate.
        start : float
            ``self.base ** start`` is the starting value of the
            sequence.
        stop : float
            ``self.base ** stop`` is the final value of the
            sequence.
        """
        if start is None:
            start = self.values.min()
        if stop is None:
            stop = self.values.max()
        return np.logspace(self._log(start), self._log(stop), num, base=self.base)

    def _log(self, data):
        data = np.log10(data)
        if self.base != 10:
            data /= np.log10(self.base)
        return data

    def __init__(self, values, scores=[0, 10], weight=1, base=10):
        Linear.__init__(self, values, scores=scores, weight=weight)
        self.base = base
        #self.values = self._log(self.values)

    def __call__(self, data):
        """
        Score the data according to this score model.

        This function takes the log of self.values, and the log of
        `data`, then performs linear interpolation onto self.scores.
        """
        # This if statement 'block' makes sure `data` is in the right
        # 'format' (object type)
        if data.__class__ is pd.Series:
            data = data.values
        return np.interp(self._log(data), self._log(self.values), self.scores)

    def __repr__(self,):
        outstr = '<ScoreLog>\nWEIGHT: %4.2f, BASE: %0.1g\n' % (self.weight, self.base)
        outstr += 'Values: %s\n' % self.values
        outstr += 'Scores: %s\n' % self.scores
        return outstr

    def __copy__(self,):
        return self.__class__(self.values, self.scores, self.weight, self.base)


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

    @property
    def shape(self, ):
        return self.table.shape

    def __len__(self, ):
        return self.table.shape[0]

    def to_array(self, col_labels=False):
        if not col_labels:
            return self.table
        shp = list(self.shape)
        shp[0] += 1
        out = np.empty(shp, dtype='O')
        out[0, :] = 'MINVAL', 'MAXVAL', 'SCORE'
        out[1:, :] = self.table[:]
        return out

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
        Score the data according to this table scorer.
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
