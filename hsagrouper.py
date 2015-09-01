from __future__ import print_function, division, absolute_import

from pprint import pprint

import numpy as np
from pandas.util.decorators import cache_readonly
from pandas.core.generic import NDFrame
from pandas.core.groupby import Grouper, BaseGrouper


class HSAGrouper(Grouper):
    """
    Custom grouper class for HSA GPU accelerated grouping
    """

    def __init__(self, column, **kwargs):
        kwargs['sort'] = True
        self.column = column

        super(HSAGrouper, self).__init__(**kwargs)

    def _get_grouper(self, obj):
        self._set_grouper(obj)

        return self._get_binner()

    def _get_binner(self):
        self.binner = MyBinner()

        # TODO: move the following logic into binner
        index = getattr(self.obj, self.column).values
        ordering = np.argsort(index)
        labels = np.unique(index)

        sorted_index = index[ordering]
        bins = []  # implicit zero
        curval = sorted_index[0]
        for i, val in enumerate(sorted_index[1:], start=1):
            if val != curval:
                bins.append(i)
                curval = val
        # Append last index
        bins.append(len(sorted_index))

        self.grouper = MyGrouper(labels, ordering, bins)

        # Sort the frame object to simplify processing
        self.obj = self.obj.reindex(ordering)
        return self.binner, self.grouper, self.obj


class MyBinner(object):
    pass


class MyGrouper(BaseGrouper):
    def __init__(self, labels, ordering, bins):
        self.binlabels = labels
        self.ordering = ordering
        self.bins = bins

    @cache_readonly
    def groups(self):
        """ dict {group name -> group labels} """
        raise NotImplementedError
        # this is mainly for compat
        # GH 3881
        result = {}
        for key, value in zip(self.binlabels, self.bins):
            if key is not tslib.NaT:
                result[key] = value
        return result

    @property
    def nkeys(self):
        raise NotImplementedError
        return 1

    def get_iterator(self, data, axis=0):
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """

        print("data", data, type(data))

        start = 0
        for label, edge in zip(self.binlabels, self.bins):
            yield label, data[start:edge]

            start = edge

        ######
        #
        # if isinstance(data, NDFrame):
        #     slicer = lambda start,edge: data._slice(slice(start,edge),axis=axis)
        #     length = len(data.axes[axis])
        # else:
        #     slicer = lambda start,edge: data[slice(start,edge)]
        #     length = len(data)
        #
        # start = 0
        # for edge, label in zip(self.bins, self.binlabels):
        #     if label is not tslib.NaT:
        #         yield label, slicer(start,edge)
        #     start = edge
        #
        # if start < length:
        #     yield self.binlabels[-1], slicer(start,None)

    def apply(self, f, data, axis=0):
        raise NotImplementedError
        result_keys = []
        result_values = []
        mutated = False
        for key, group in self.get_iterator(data, axis=axis):
            object.__setattr__(group, 'name', key)

            # group might be modified
            group_axes = _get_axes(group)
            res = f(group)

            if not _is_indexed_like(res, group_axes):
                mutated = True

            result_keys.append(key)
            result_values.append(res)

        return result_keys, result_values, mutated

    @cache_readonly
    def indices(self):
        raise NotImplementedError
        indices = collections.defaultdict(list)

        i = 0
        for label, bin in zip(self.binlabels, self.bins):
            if i < bin:
                if label is not tslib.NaT:
                    indices[label] = list(range(i, bin))
                i = bin
        return indices

    @cache_readonly
    def group_info(self):
        raise NotImplementedError
        ngroups = self.ngroups
        obs_group_ids = np.arange(ngroups)
        comp_ids = np.repeat(np.arange(ngroups), np.diff(np.r_[0, self.bins]))
        return comp_ids, obs_group_ids, ngroups

    @cache_readonly
    def ngroups(self):
        raise NotImplementedError
        return len(self.binlabels)

    @cache_readonly
    def result_index(self):
        raise NotImplementedError
        mask = self.binlabels.asi8 == tslib.iNaT
        return self.binlabels[~mask]

    @property
    def levels(self):
        raise NotImplementedError
        return [self.binlabels]

    @property
    def names(self):
        raise NotImplementedError
        return [self.binlabels.name]

    @property
    def groupings(self):
        # for compat
        raise NotImplementedError
        return None

    def size(self):
        """
        Compute group sizes

        """
        raise NotImplementedError
        index = self.result_index
        base = Series(np.zeros(len(index), dtype=np.int64), index=index)
        indices = self.indices
        for k, v in compat.iteritems(indices):
            indices[k] = len(v)
        bin_counts = Series(indices, dtype=np.int64)
        # make bin_counts.index to have same name to preserve it
        bin_counts.index.name = index.name
        result = base.add(bin_counts, fill_value=0)
        # addition with fill_value changes dtype to float64
        result = result.astype(np.int64)
        return result

    # ----------------------------------------------------------------------
    # cython aggregation
    #
    # _cython_functions = {
    #     'add': 'group_add_bin',
    #     'prod': 'group_prod_bin',
    #     'mean': 'group_mean_bin',
    #     'min': 'group_min_bin',
    #     'max': 'group_max_bin',
    #     'var': 'group_var_bin',
    #     'ohlc': 'group_ohlc',
    #     'first': {
    #         'name': 'group_nth_bin',
    #         'f': lambda func, a, b, c, d: func(a, b, c, d, 1)
    #     },
    #     'last': 'group_last_bin',
    #     'count': 'group_count_bin',
    # }
    #
    # _name_functions = {
    #     'ohlc': lambda *args: ['open', 'high', 'low', 'close']
    # }

    def _aggregate(self, result, counts, values, agg_func, is_numeric=True):
        raise NotImplementedError
        if values.ndim > 3:
            # punting for now
            raise NotImplementedError("number of dimensions is currently "
                                      "limited to 3")
        elif values.ndim > 2:
            for i, chunk in enumerate(values.transpose(2, 0, 1)):
                agg_func(result[:, :, i], counts, chunk, self.bins)
        else:
            agg_func(result, counts, values, self.bins)

        return result

    def agg_series(self, obj, func):
        raise NotImplementedError
        dummy = obj[:0]
        grouper = lib.SeriesBinGrouper(obj, func, self.bins, dummy)
        return grouper.get_result()
