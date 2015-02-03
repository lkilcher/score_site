from copy import deepcopy
import pandas as pd
import numpy as np


class HotSpotData(object):

    def __init__(self, data, resdata):
        self.data = data
        self.resdata = resdata

    def __getitem__(self, val):
        out = self.data.loc[val]
        out.resdata = self.resdata[self.resdata.name == val]
        out.set_value('max_resource', max(out.resdata.resource))
        return out

    def to_excel(self, fname):
        if basestring in fname.__class__.__mro__:
            fname = pd.io.excel.ExcelWriter(fname)
        self.data.to_excel(fname, sheet_name='SiteData')
        self.resdata.to_excel(fname, sheet_name='ResourceData', index=False)
        fname.close()

    def to_csv(self, fname):
        fname = fname.rstrip('.csv')
        self.data.to_csv(fname + '.csv',)
        self.resdata.to_csv(fname + '_res.csv', index=False)

    def __repr__(self,):
        return self.data.__repr__()

    def __copy__(self):
        return HotSpotData(self.data.copy(), self.resdata.copy())

    copy = __copy__

    def rank(self, clear_0_nan=True):
        bdi = np.isnan(self.data['score_total']) | (self.data['score_total'] == 0)
        resbdi = np.isnan(self.resdata['score_total'])
        if clear_0_nan:
            self.data = self.data[~bdi]
            self.resdata = self.resdata[~resbdi]
        else:
            self.data['score_total'][bdi] = -1
            self.resdata['score_total'][resbdi] = -1
        # Sort the results:
        self.data = self.data.sort(['score_total', 'load'],
                                   ascending=[False, False])
        self.resdata = self.resdata.sort(['name', 'score_total', 'resource'],
                                         ascending=[True, False, False])
        if not clear_0_nan:
            self.data['score_total'][bdi] = np.NaN
            self.resdata['score_total'][resbdi] = np.NaN


class SumModel(object):

    def __repr__(self,):
        outstr = "'%s' site scoring %s:\n" % (
            self.tag,
            str(self.__class__).rstrip("'>").split('.')[-1],
            )
        outstr += self.tables.__repr__()
        return outstr

    def __init__(self, tag=None, **kwargs):
        self.tables = deepcopy(kwargs)
        self.resource_vars = list(set(self.tables.keys()) &
                                  {'dist', 'resource', 'depth'})
        self.tag = tag

    ## def __getitem__(self, val):
    ##     if val.__class__ in [tuple, list]:
    ##         return self.__class__(**{nm:self.tables[nm] for nm in val if nm in self.tables})
    ##     else:
    ##         return self.tables[val]

    def __copy__(self, model_class=None, tag=None, **kwargs):
        if model_class is None:
            model_class = self.__class__
        tables = deepcopy(self.tables)
        tables.update(**kwargs)
        return model_class(tag=tag, **tables)

    copy = __copy__

    def _norm_weights(self, subset=None):
        wnorm = 0.
        nms = self.tables.keys()
        if subset is not None:
            nms = [nm for nm in nms if nm in subset]
        for nm in nms:
            wnorm += self.tables[nm].weight
        out = {}
        for nm in nms:
            out[nm] = self.tables[nm].weight / wnorm
        return out

    def _calc_total(self, data, weights):
        score = pd.Series(np.zeros(len(data.index)),
                          index=data.index)
        for nm, w in weights.iteritems():
            score += data['score_' + nm] * weights[nm]
        return score

    def _score_it(self, data, names=None):
        """
        Score the `data` according to the tables in this model.
        """
        weights = self._norm_weights(names)
        for nm in weights:
            # Score the resource data:
            data['score_' + nm] = self.tables[nm](data[nm])
        data['score_total'] = self._calc_total(data, weights)
        return data

    def _assign_resource2site(self, out):
        """
        Add columns to the site data that contain the highest-scoring
        resource data.
        """
        for nm in self.resource_vars:
            # Initialize the columns of the site data:
            out.data[nm] = pd.Series(np.zeros(len(out.data.index)),
                                     index=out.data.index)
        for site in out.data.index:
            inds = (out.resdata.name == site)
            if inds.sum() == 0 or np.isnan(out.resdata['score_total'][inds]).all():
                idx = None
            else:
                idx = np.argmax(out.resdata['score_total'][inds])
            for nm in self.resource_vars:
                if idx is None:
                    val = np.NaN
                else:
                    val = out.resdata[nm][idx]
                out.data[nm][site] = val

    def __call__(self, data):
        """
        Score the `data` according to this model.
        """
        out = data.copy()
        # Calculate the resource data scores:
        self._score_it(out.resdata, self.resource_vars)
        # Assign the highest ranking resource data to the site data:
        self._assign_resource2site(out)
        # Now score the site data:
        self._score_it(out.data)
        return out


class ProdModel(SumModel):

    def _calc_total(self, data, weights):
        score = pd.Series(np.ones(len(data.index)),
                          index=data.index)
        for nm, w in weights.iteritems():
            score *= data['score_' + nm] ** weights[nm]
        return score


class MultiModel(object):

    """
    A class for computing scores for several models, and selecting
    data from the highest scoring one.
    """

    def __init__(self, *args):
        self.models = args

    def __call__(self, data):
        tag = pd.Series(np.empty(len(data.data.index), dtype='S20'),
                        index=data.data.index, )
        restag = pd.Series(np.empty(len(data.resdata.index), dtype='S20'),
                           index=data.resdata.index, )
        tag[:] = self.models[0].tag
        restag[:] = self.models[0].tag
        for idx, mdl in enumerate(self.models):
            now = mdl(data)
            if idx == 0:
                out = now
            else:
                inds = now.data['score_total'] > out.data['score_total']
                out.data[inds] = now.data[inds]
                tag[inds] = mdl.tag
                inds = now.resdata['score_total'] > out.resdata['score_total']
                out.resdata[inds] = now.resdata[inds]
                restag[inds] = mdl.tag
        out.data['best_model'] = tag
        out.resdata['best_model'] = restag

        return out


def score_data(data, spec):
    """
    This function uses the score table (or array) `spec` to calculate
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
    for mn, mx, score in spec:
        # Test that data is within the range, and assign scores:
        out[((mn < data) & (data <= mx))] = score
    return out  # Return the output data from the function.


class ScoreTable(object):

    def __init__(self, table, weight=1):
        if np.array not in table.__class__.__mro__:
            self.table = np.array(table)
        self.weight = weight

    def __copy__(self,):
        return self.__class__(self.table.copy(), weight=self.weight)

    copy = __copy__

    def __repr__(self,):
        outstr = 'WEIGHT: %4.2f\n------ Range --------- :  Score\n' % (self.weight, )
        for v in self.table:
            outstr += ' %9.1f - %9.1f : %5.1f\n' % tuple(v)
        return outstr

    def __call__(self, data):
        return score_data(data, self.table)

    def sumweight(self, data):
        return self(data) * self.weight

    def prodweight(self, data):
        return self(data) ** self.weight
