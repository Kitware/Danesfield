import json, os, unittest
import gdal
import pdal
import gaia
import geopandas
import numpy as np


class TestGaiaInstall(unittest.TestCase):

    def test_read_file(self):
        fname = os.path.join(os.environ['RECIPE_DIR'], 'test_data', 'test.shp')
        df = geopandas.read_file(fname)
        bounds = df.total_bounds
        self.assertEqual(len(bounds), 4)

        # >>> df.total_bounds
        # array([-82.83,  25.35, -75.52,  36.18])

        self.assertGreater(bounds[0], -83)
        self.assertLess(bounds[0], -82)

        self.assertGreater(bounds[1], 25)
        self.assertLess(bounds[1], 26)

        self.assertGreater(bounds[2], -76)
        self.assertLess(bounds[2], -75)

        self.assertGreater(bounds[3], 36)
        self.assertLess(bounds[3], 37)

    def test_gdal(self):
        fileName = os.path.join(os.environ['RECIPE_DIR'], 'test_data', 'tasmax-1-layer-cropped.tif')

        dataset = gdal.Open(fileName)

        metadata = dataset.GetMetadata()

        self.assertEqual(dataset.RasterXSize, 50)
        self.assertEqual(dataset.RasterYSize, 50)
        self.assertEqual(metadata['tasmax#standard_name'], 'air_temperature')

        sumArray = np.zeros((dataset.RasterYSize, dataset.RasterXSize))
        total = 0
        count = 0
        numBands = dataset.RasterCount

        self.assertEqual(numBands, 1)

        for bandId in range(numBands):
            band = dataset.GetRasterBand(bandId + 1).ReadAsArray()
            sumArray += band

        sumArray /= numBands
        total = np.sum(np.sum(sumArray))
        count = sumArray.size
        avgCell = total / count
        minCell = np.min(sumArray)
        maxCell = np.max(sumArray)

        self.assertEqual(count, 2500)
        self.assertLess(minCell - 280, 1)
        self.assertLess(maxCell - 304, 1)
        self.assertLess(avgCell - 295, 1)

    def test_pdal(self):
        fname = os.path.join(os.environ['RECIPE_DIR'], 'test_data', '1.2-with-color.las')

        jsonPipeline = """
        {
          "pipeline": [
            "%s",
            {
                "type": "filters.sort",
                "dimension": "X"
            }
          ]
        }""" % fname

        pipeline = pdal.Pipeline(jsonPipeline)
        pipelineValid = pipeline.validate()

        self.assertEqual(pipelineValid, True)

        count = pipeline.execute()

        self.assertEqual(count, 1065)

        # Could do more checks with the values below
        # arrays = pipeline.arrays
        # metadata = pipeline.metadata

        # md = json.loads(metadata)

if __name__ == '__main__':
    unittest.main()
