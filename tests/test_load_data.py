from datacube import Datacube
from datacube.api.query import query_group_by
import numpy as np
from collections import Sequence
from types import SimpleNamespace
import pytest

from pathlib import Path
from datacube.testutils import (
    mk_sample_dataset,
    mk_test_image,
)
from datacube.testutils.io import write_gtiff, rio_slurp
from datacube.testutils.iodriver import NetCDF
from datacube.utils import ignore_exceptions_if


def gen_tiff_dataset(bands,
                     base_folder,
                     prefix='',
                     timestamp='2018-07-19',
                     **kwargs):
    """
       each band:
         .name    - string
         .values  - ndarray
         .nodata  - numeric|None

    :returns:  (Dataset, GeoBox)
    """
    if not isinstance(bands, Sequence):
        bands = (bands,)

    # write arrays to disk and construct compatible measurement definitions
    gbox = None
    mm = []
    for band in bands:
        name = band.name
        fname = prefix + name + '.tiff'
        meta = write_gtiff(base_folder/fname, band.values,
                           nodata=band.nodata,
                           overwrite=True,
                           **kwargs)

        gbox = meta.gbox

        mm.append(dict(name=name,
                       path=fname,
                       layer=1,
                       dtype=meta.dtype))

    uri = Path(base_folder/'metadata.yaml').absolute().as_uri()
    ds = mk_sample_dataset(mm, uri=uri, timestamp=timestamp)
    return ds, gbox


def test_load_data(tmpdir):
    tmpdir = Path(str(tmpdir))

    group_by = query_group_by('time')
    spatial = dict(resolution=(15, -15),
                   offset=(11230, 1381110),)

    nodata = -999
    aa = mk_test_image(96, 64, 'int16', nodata=nodata)

    ds, gbox = gen_tiff_dataset([SimpleNamespace(name='aa', values=aa, nodata=nodata)],
                                tmpdir,
                                prefix='ds1-',
                                timestamp='2018-07-19',
                                **spatial)
    assert ds.time is not None

    ds2, _ = gen_tiff_dataset([SimpleNamespace(name='aa', values=aa, nodata=nodata)],
                              tmpdir,
                              prefix='ds2-',
                              timestamp='2018-07-19',
                              **spatial)
    assert ds.time is not None
    assert ds.time == ds2.time

    sources = Datacube.group_datasets([ds], 'time')
    sources2 = Datacube.group_datasets([ds, ds2], group_by)

    mm = ['aa']
    mm = [ds.type.measurements[k] for k in mm]

    ds_data = Datacube.load_data(sources, gbox, mm)
    assert ds_data.aa.nodata == nodata
    np.testing.assert_array_equal(aa, ds_data.aa.values[0])

    custom_fuser_call_count = 0

    def custom_fuser(dest, delta):
        nonlocal custom_fuser_call_count
        custom_fuser_call_count += 1
        dest[:] += delta

    progress_call_data = []

    def progress_cbk(n, nt):
        progress_call_data.append((n, nt))

    ds_data = Datacube.load_data(sources2, gbox, mm, fuse_func=custom_fuser,
                                 progress_cbk=progress_cbk)
    assert ds_data.aa.nodata == nodata
    assert custom_fuser_call_count > 0
    np.testing.assert_array_equal(nodata + aa + aa, ds_data.aa.values[0])

    assert progress_call_data == [(1, 2), (2, 2)]


def test_hdf5_lock_release_on_failure():
    from datacube.storage._rio import RasterDatasetDataSource, _HDF5_LOCK
    from datacube.storage import BandInfo

    band = dict(name='xx',
                layer='xx',
                dtype='uint8',
                units='K',
                nodata=33)

    ds = mk_sample_dataset([band],
                           uri='file:///tmp/this_probably_doesnot_exist_37237827513/xx.nc',
                           format=NetCDF)
    src = RasterDatasetDataSource(BandInfo(ds, 'xx'))

    with pytest.raises(OSError):
        with src.open():
            assert False and "Did not expect to get here"

    assert not _HDF5_LOCK._is_owned()


def test_rio_slurp(tmpdir):
    w, h, dtype, nodata, ndw = 96, 64, 'int16', -999, 7

    pp = Path(str(tmpdir))

    aa = mk_test_image(w, h, dtype, nodata, nodata_width=ndw)

    assert aa.shape == (h, w)
    assert aa.dtype.name == dtype
    assert aa[10, 30] == (30 << 8) | 10
    assert aa[10, 11] == nodata

    aa0 = aa.copy()
    mm0 = write_gtiff(pp/"rio-slurp-aa.tif", aa, nodata=-999, overwrite=True)
    mm00 = write_gtiff(pp/"rio-slurp-aa-missing-nodata.tif", aa, nodata=None, overwrite=True)

    aa, mm = rio_slurp(mm0.path)
    np.testing.assert_array_equal(aa, aa0)
    assert mm.gbox == mm0.gbox
    assert aa.shape == mm.gbox.shape

    aa, mm = rio_slurp(mm0.path, aa0.shape)
    np.testing.assert_array_equal(aa, aa0)
    assert aa.shape == mm.gbox.shape
    assert mm.gbox is mm.src_gbox

    aa, mm = rio_slurp(mm0.path, (3, 7))
    assert aa.shape == (3, 7)
    assert aa.shape == mm.gbox.shape
    assert mm.gbox != mm.src_gbox
    assert mm.src_gbox == mm0.gbox
    assert mm.gbox.extent == mm0.gbox.extent

    aa, mm = rio_slurp(mm0.path, aa0.shape)
    np.testing.assert_array_equal(aa, aa0)
    assert aa.shape == mm.gbox.shape

    aa, mm = rio_slurp(mm0.path, mm0.gbox, resampling='nearest')
    np.testing.assert_array_equal(aa, aa0)

    aa, mm = rio_slurp(mm0.path, gbox=mm0.gbox, dtype='float32')
    assert aa.dtype == 'float32'
    np.testing.assert_array_equal(aa, aa0.astype('float32'))

    aa, mm = rio_slurp(mm0.path, mm0.gbox, dst_nodata=-33)
    np.testing.assert_array_equal(aa == -33, aa0 == -999)

    aa, mm = rio_slurp(mm00.path, mm00.gbox, dst_nodata=None)
    np.testing.assert_array_equal(aa, aa0)


def test_rio_slurp_with_gbox(tmpdir):
    w, h, dtype, nodata, ndw = 96, 64, 'int16', -999, 7

    pp = Path(str(tmpdir))
    aa = mk_test_image(w, h, dtype, nodata, nodata_width=ndw)
    assert aa.dtype.name == dtype
    assert aa[10, 30] == (30 << 8) | 10
    assert aa[10, 11] == nodata

    aa = np.stack([aa, aa[::-1, ::-1]])
    assert aa.shape == (2, h, w)
    aa0 = aa.copy()

    mm = write_gtiff(pp/"rio-slurp-aa.tif", aa, nodata=-999, overwrite=True)
    assert mm.count == 2

    aa, mm = rio_slurp(mm.path, mm.gbox)
    assert aa.shape == aa0.shape
    np.testing.assert_array_equal(aa, aa0)


def test_missing_file_handling():
    with pytest.raises(IOError):
        rio_slurp('no-such-file.tiff')

    # by default should catch any exception
    with ignore_exceptions_if(True):
        rio_slurp('no-such-file.tiff')

    # this is equivalent to previous default behaviour, note that missing http
    # resources are not OSError
    with ignore_exceptions_if(True, (OSError,)):
        rio_slurp('no-such-file.tiff')

    # check that only requested exceptions are caught
    with pytest.raises(IOError):
        with ignore_exceptions_if(True, (ValueError, ArithmeticError)):
            rio_slurp('no-such-file.tiff')
