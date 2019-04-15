import xarray
import numpy as np
from datacube.model import Measurement
from .impl import Transformation
from osgeo import ogr
from osgeo import gdal
from osgeo import osr

class MaskByValue(Transformation):
    '''
    '''
    def __init__(self, mask_measurement_name, greater_than=None, smaller_than=None):
        self.greater_than = greater_than
        self.smaller_than = smaller_than
        if self.greater_than is not None and self.smaller_than is not None:
            if self.greater_than > self.smaller_than:
                raise("greater_than should smaller than smaller_than")
        self.mask_measurement_name = mask_measurement_name
            
    def compute(self, data):
        if self.greater_than is not None:
            results = data[self.mask_measurement_name].where(data[self.mask_measurement_name] > self.greater_than, -9999)
        else:
            results = data[self.mask_measurement_name]

        if self.smaller_than is not None:
            results = results.where(results < self.smaller_than, -9999)

        results = results > -9999 
        results.attrs['crs'] = data.attrs['crs']
        return results

    def measurements(self, input_measurements):
        if self.mask_measurement_name not in list(input_measurements.keys()):
            raise("have to mask by the band in product")

        return {self.mask_measurement_name: Measurement(name=self.mask_measurement_name, dtype='bool', nodata=0, units=1)}

class TCIndex(Transformation):
    '''
    '''
    def __init__(self, category='wetness', coeffs=None):
        self.category = category
        if coeffs is None:
            self.coeffs = {
                'brightness': {'blue': 0.2043, 'green': 0.4158, 'red': 0.5524, 'nir': 0.5741,
                                'swir1': 0.3124, 'swir2': 0.2303},
                 'greenness': {'blue': -0.1603, 'green': -0.2819, 'red': -0.4934, 'nir': 0.7940,
                 'swir1': -0.0002, 'swir2': -0.1446},
                 'wetness': {'blue': 0.0315, 'green': 0.2021, 'red': 0.3102, 'nir': 0.1594,
                 'swir1': -0.6806, 'swir2': -0.6109}
            }
        else:
            self.coeffs = coeffs
        self.var_name = f'TC{category[0].upper()}'

    def compute(self, data):
        tci_var = []
        for var in data.data_vars:
            nodata = getattr(data[var], 'nodata', -1)
            data[var] = data[var].where(data[var] > nodata)
            tci_var.append(data[var] * self.coeffs[self.category][var])
        tci_var = sum(tci_var)
        tci_var.values[np.isnan(tci_var.values)] = -9999
        tci = xarray.Dataset(data_vars={self.var_name: tci_var.astype('float32')},
                             coords=data.coords,
                             attrs=dict(crs=data.crs))
        tci[self.var_name].attrs['nodata'] = -9999
        tci[self.var_name].attrs['units'] = 1 
        return tci
        
    def measurements(self, input_measurements):
        return {self.var_name: Measurement(name=self.var_name, dtype='float32', nodata=-9999, units='1')}



class MangroveCC(Transformation):
    def __init__(self, thresholds, shape_file, bands=None):
        self.thresholds = thresholds
        if bands is None:
            self.bands = ['extent', 'canopy_cover_class']
        else:
            self.bands = bands
        self.shape_file = shape_file

    def measurements(self, input_measurements):
        output_measurements = dict()
        for band in self.bands:
            output_measurements[band] = Measurement(name=band, dtype='int16', nodata=-1, units='1')
        return output_measurements

    def compute(self, data):
        var_name = list(data.data_vars.keys())[0]

        rast_data = data[var_name].where(data[var_name] > 0, -9999)
        print('before raster', np.where(rast_data.data == -9999))
        rast_data = rast_data.where(self.generate_rasterize(data[var_name]) == 1, -1)
        print('after raster', np.where(rast_data.data == -9999))

        cover_extent = rast_data.copy(True)
        cover_extent.data = np.zeros(cover_extent.shape, dtype='int16') - 1  
        cover_extent.data[rast_data.data > self.thresholds[0]] = 1
        cover_extent.data[rast_data.data == -9999] = 0 

        cover_type = rast_data.copy(True)
        cover_type.data = np.zeros(cover_type.shape, dtype='int16') - 1
        level_threshold = 1
        for s_t in self.thresholds:
            cover_type.data[rast_data.data > s_t] = level_threshold
            level_threshold += 1
        cover_type.data[rast_data.data == -9999] = 0 

        outputs = {}
        outputs[self.bands[0]] = cover_extent
        outputs[self.bands[0]].attrs['nodata'] = -1 
        outputs[self.bands[0]].attrs['units'] = 1 
        outputs[self.bands[1]] = cover_type
        outputs[self.bands[1]].attrs['nodata'] = -1 
        outputs[self.bands[1]].attrs['units'] = 1 
        return xarray.Dataset(outputs, attrs=dict(crs=data.crs))

    def generate_rasterize(self, data):
        source_ds = ogr.Open(self.shape_file)
        source_layer = source_ds.GetLayer()

        yt, xt = data[0].shape
        xres = 25
        yres = -25
        no_data = 0

        xcoord = data.coords['x'].min()
        ycoord = data.coords['y'].max()
        geotransform = (xcoord - (xres*0.5), xres, 0, ycoord - (yres*0.5), 0, yres)

        target_ds = gdal.GetDriverByName('MEM').Create('', xt, yt, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geotransform)
        albers = osr.SpatialReference()
        albers.ImportFromEPSG(3577)
        target_ds.SetProjection(albers.ExportToWkt())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(no_data)

        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
        return band.ReadAsArray()
