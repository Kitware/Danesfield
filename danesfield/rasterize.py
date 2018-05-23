import subprocess

import gdal
import numpy
from scipy.ndimage import morphology

from . import gdal_utils


ELEVATED_ROADS_QUERY = (
    "bridge = 1 and ("
    "    type = 'monorail'"
    "    or \"class\" = 'highway'"
    "        and type not in ('footway', 'pedestrian')"
    ")"
)


def rasterize_file(
        vector_filename_in, reference_file, thin_line_raster_filename_out,
        dilation_structure=None, dilation_iterations=1, query=None,
):
    """
    Rasterize the vector geometry at vector_filename_in, returning an
    ndarray.  Use the image dimensions, boundary, and other metadata
    from reference_file (an in-memory object).  A rasterization with
    only 1px-thick lines is written to thin_line_raster_filename_out.

    The lines are thickened with binary dilation, with the structuring
    element and iteration count provided by the dilation_structure and
    dilation_iterations arguments respectively.

    If query is passed, it is used as a SQL where-clause to select
    certain features.
    """
    rasterize_file_thin_line(vector_filename_in, reference_file,
                             thin_line_raster_filename_out, query)
    thin_line_file = gdal_utils.gdal_open(thin_line_raster_filename_out)
    return morphology.binary_dilation(
        thin_line_file.GetRasterBand(1).ReadAsArray(),
        dilation_structure, dilation_iterations,
    )


def rasterize_file_thin_line(vector_filename_in, reference_file,
                             raster_filename_out, query=None):
    """
    Rasterize the vector geometry at vector_filename_in to a file at
    raster_filename_out.  Get image dimensions, boundary, and other
    metadata from reference_file (an in-memory object).  Note that the
    lines in the output file are only 1px thick.

    If query is passed, use it as a SQL where-clause to select certain
    features.
    """
    size = reference_file.RasterYSize, reference_file.RasterXSize
    gdal_utils.gdal_save(numpy.zeros(size, dtype=numpy.uint8),
                         reference_file, raster_filename_out, gdal.GDT_Byte)
    subprocess.run(['gdal_rasterize', '-burn', '1']
                   + ([] if query is None else ['-where', query])
                   + [vector_filename_in, raster_filename_out],
                   check=True,
                   stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)
