"""
This package contains the scoring models. These define the method for
aggregating scores. A single scoring 'model' is initialized with a set
of several 'scorer' objects from the :mod:`.scorers` module, e.g.::

    a_scoring_model = SumModel(dist=scorers.Linear([0, 30], score=[0, 10]),
                               resource=scorers.Linear([10, 30], score=[0, 10]) )

When a scoring model is called to operate on a
:class:`.HotSpotCollection`, e.g.::

    results = a_scoring_model(a_HotSpotCollection_instance)

it calls each scorer it contains (the individual scores are stored in
`score_<name>` where `<name>` is the field being scored), and computes
an aggregate score (and stores it in 'score_total').

"""
from copy import deepcopy
import pandas as pd
import numpy as np
from base import cmaps


class SumModel(object):
    """
    A 'sum-method' scoring model.

    Parameters
    ----------
    tag : string or None
          A string to attach to this scoring model.
    **kwargs : key-value pairs
               Keys must match the column of the data you wish to
               score. The values are the :mod:`.scorers` to apply to
               that column.

    Notes
    -----

    The sum-method aggregates scores according to:

    .. math::

        S_{tot} = \sum_i w_i \cdot S_i

    where :math:`S_i` and `w_i` are the scores and weights for each of
    the data columns.


    This model automatically normalizes the weights so that
    :math:`\sum_i w_i = 1`. This means that :math:`S_{tot}` will be
    between 0 and 10.

    """

    def to_html(self,
                buf,
                legend_vals={},
                cmap=cmaps['green'],
                hline_widths=[1],
                score_col=True,
                ):
        """
        Write the scorers in this model as a set of html tables
        colored by cmap according to the scores.

        Parameters
        ----------
        buf : string or file_buffer
              The file to write the table to. If it is a string, and
              it does not end in '.htm' or '.html', the latter will be
              appended to the filename.
        legend_vals : dict
                      The keys should match those in the scorers of
                      this model. Each value should be an array_like
                      of values at which the table should have entries.
        cmap : <matplotlib colormap>
               The colormap to color the tables with.
        hline_widths : iterable
                       A list of the line widths to use.
        score_col : bool (default: True)
                    Specify whether to include the 'score' column in
                    the output.
        """
        with open(buf, 'w') as fl:
            for key, scr in self.scorers.iteritems():
                scr.to_html(fl,
                            legend_vals=legend_vals.get(key, None),
                            title=key,
                            cmap=cmap,
                            hline_widths=hline_widths,
                            score_col=score_col,
                            )

    def __repr__(self,):
        outstr = "'%s' site scoring %s:\n" % (
            self.tag,
            str(self.__class__).rstrip("'>").split('.')[-1],
        )
        outstr += self.scorers.__repr__()
        return outstr

    def __init__(self, tag=None, zero_1score_zero=False, **kwargs):
        self.scorers = deepcopy(kwargs)
        self.tag = tag
        self.zero_1score_zero = zero_1score_zero

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
        zero_1col = np.zeros(len(data.index), dtype='bool')
        for nm, w in weights.iteritems():
            score += data['score_' + nm] * weights[nm]
            zero_1col |= data['score_' + nm] == 0
        if self.zero_1score_zero:
            score[zero_1col] = 0
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

    def _set_resource_vars(self, data):
        data.resource_vars = []
        if data.resdata is None:
            return
        for nm in self.scorers:
            if (nm not in data.resdata.columns) and (nm not in data.data.columns):
                raise Exception("The data has no column '%s'" % nm)
            if (nm in data.resdata.columns) and (nm not in data.data.columns):
                data.resource_vars += [nm]

    def _assign_resource2site(self, out):
        """
        Add columns to the site data that contain the highest-scoring
        resource data.
        """
        for nm in out.resource_vars:
            # Initialize the columns of the site data:
            out.data[nm] = pd.Series(np.zeros(len(out.data.index)),
                                     index=out.data.index)
        for site in out.data.index:
            inds = (out.resdata.name == site)
            if inds.sum() == 0 or np.isnan(out.resdata['score_total'][inds]).all():
                idx = None
            else:
                idx = np.argmax(out.resdata['score_total'][inds])
            for nm in out.resource_vars:
                if idx is None:
                    val = np.NaN
                else:
                    val = out.resdata[nm][idx]
                out.data.loc[site, nm] = val

    def __call__(self, data):
        """
        Score `data` according to this model.
        """
        out = data.copy()
        self._set_resource_vars(out)
        out.model = self
        if out.resdata is not None:
            # Calculate the resource data scores:
            self._score_it(out.resdata, out.resource_vars)
            # Assign the highest ranking resource data to the site data:
            self._assign_resource2site(out)
        # Now score the site data:
        self._score_it(out.data)
        return out


class ProdModel(SumModel):
    """
    A 'product-method' scoring model.

    Parameters
    ----------
    tag : string or None
          A string to attach to this scoring model.
    **kwargs : key-value pairs
               Keys must match the column of the data you wish to
               score. The values are the :mod:`.scorers` to apply to
               that column.

    Notes
    -----

    The product-method aggregates scores according to:

    .. math::

        S_{tot} = \Pi_i S_i^w_i

    where :math:`S_i` and `w_i` are the scores and weights for each of
    the data columns.

    This model automatically normalizes the weights so that
    :math:`\sum_i w_i = 1`. This means that :math:`S_{tot}` will be
    between 0 and 10.

    """

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
        """
        Score `data` according to this MultiModel.
        """
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
