import xarray
import numpy as np
import pandas as pd

from collections import Sequence
from functools import partial
from .impl import VirtualProductException, Transformation, Measurement, VirtualDatasetBox
from .stat_funcs import argpercentile, anynan, axisindex

class Percentile(Transformation):
    """
    Per-band percentiles of observations through time.
    The different percentiles are stored in the output as separate bands.
    The q-th percentile of a band is named `{band}_PC_{q}`.

    :param q: list of percentiles to compute
    :param per_pixel_metadata: provenance metadata to attach to each pixel
    :arg minimum_valid_observations: if not enough observations are available,
                                     percentile will return `nodata`
    """

    def __init__(self, q,
                 minimum_valid_observations=0,
                 not_valid_mark=None):

        if isinstance(q, Sequence):
            self.qs = q
        else:
            self.qs = [q]

        self.minimum_valid_observations = minimum_valid_observations
        self.not_valid_mark = not_valid_mark

    def compute(self, data):
        # calculate masks for pixel without enough data
        for var in data.data_vars:
            nodata = getattr(data[var], 'nodata', None)
            if nodata is not None:
                data[var].attrs['dtype'] = data[var].dtype
                data[var] = data[var].where(data[var] > nodata)
        not_enough  = np.logical_and(data.count(dim='time') < self.minimum_valid_observations, data.count(dim='time') > 0)

        def single(q):
            stat_func = partial(xarray.Dataset.reduce, dim='time', keep_attrs=True,
                                func=argpercentile, q=q)
            result = stat_func(data)

            def index_dataset(var): 
                return axisindex(data[var.name].values, var.values) 

            result = result.apply(index_dataset, keep_attrs=True)

            def mask_not_enough(var):
                nodata = getattr(data[var.name], 'nodata', -1)
                if self.not_valid_mark is not None:
                    var.values[not_enough[var.name]] = self.not_valid_mark
                else:
                    var.values[not_enough[var.name]] = nodata
                var.values[np.isnan(var.values)] = nodata
                var.attrs['nodata'] = nodata
                if data[var.name].attrs['dtype'] == 'int8':
                    data_type = 'int16'
                else:
                    data_type = data[var.name].attrs['dtype']
                var = var.astype(data_type)
                return var

            return result.apply(mask_not_enough, keep_attrs=True).rename({var: var + '_PC_' + str(q) for var in result.data_vars})

        result = xarray.merge(single(q) for q in self.qs)
        result.attrs['crs'] = data.attrs['crs']
        return result

    def measurements(self, input_measurements):
        renamed = dict()
        for key, m in input_measurements.items():
            for q in self.qs:
                renamed[key + '_PC_' + str(q)] = Measurement(**{**m, 'name': key + '_PC_' + str(q)})
        return renamed
