from copy import deepcopy
import pandas as pd
import numpy as np


class SumModel(object):

    def __repr__(self,):
        outstr = "'%s' site scoring %s:\n" % (
            self.tag,
            str(self.__class__).rstrip("'>").split('.')[-1],
            )
        outstr += self.scorers.__repr__()
        return outstr

    def __init__(self, tag=None, **kwargs):
        self.scorers = deepcopy(kwargs)
        self.resource_vars = list(set(self.scorers.keys()) &
                                  {'dist', 'resource', 'depth'})
        self.tag = tag

    def __copy__(self, model_class=None, tag=None, **kwargs):
        if model_class is None:
            model_class = self.__class__
        scorers = deepcopy(self.scorers)
        scorers.update(**kwargs)
        return model_class(tag=tag, **scorers)

    copy = __copy__

    def _norm_weights(self, subset=None):
        wnorm = 0.
        nms = self.scorers.keys()
        if subset is not None:
            nms = [nm for nm in nms if nm in subset]
        for nm in nms:
            wnorm += self.scorers[nm].weight
        out = {}
        for nm in nms:
            out[nm] = self.scorers[nm].weight / wnorm
        return out

    def _calc_total(self, data, weights):
        score = pd.Series(np.zeros(len(data.index)),
                          index=data.index)
        for nm, w in weights.iteritems():
            score += data['score_' + nm] * weights[nm]
        return score

    def _score_it(self, data, names=None):
        """
        Score the `data` according to the scorers in this model.
        """
        weights = self._norm_weights(names)
        for nm in weights:
            # Score the resource data:
            data['score_' + nm] = self.scorers[nm](data[nm])
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
                out.data.loc[site, nm] = val

    def __call__(self, data):
        """
        Score the `data` according to this model.
        """
        out = data.copy()
        out.model = self
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
