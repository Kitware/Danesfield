#!/bin/bash

tiler()
{
    PYTHONPATH=. ~/projects/VTK/build/bin/vtkpython tools/tiler.py "$@"
}

print_parameters ()
{
    echo "$0 -c[|--city] jacksonville|berlin|nyc -k[--keep-intermediate-files]"
    echo "   -g[--gltf-conversions-only]"
    echo "-c <city>: selects the city mesh to convert to 3D Tiles"
}

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -c|--city)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        CITY=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      print_parameters "$0"
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

# generate 3D Tiles
cd ~/projects/danesfield/danesfield || exit

if [ "${CITY}" = "jacksonville" ]; then
    dir=jacksonville-3d-tiles
    rm -rf ${dir}
    mkdir ${dir}
    tiler ../../../data/CORE3D/Jacksonville/building_*building*.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -b 100 -n 2 --dont_save_textures --dont_convert_gltf
elif [ "${CITY}" = "test-jacksonville" ]; then
    dir=jacksonville-3d-tiles
    rm -rf ${dir}
    mkdir ${dir}
    tiler ../../../data/CORE3D/Jacksonville/triangle.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -b 2 --dont_save_textures
elif [ "${CITY}" = "berlin" ]; then
    dir=berlin-3d-tiles
    rm -rf $dir
    mkdir $dir
    tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o ${dir} --utm_zone 33 --utm_hemisphere N -b 200 --dont_save_textures --number_of_buildings 10000
elif [ "${CITY}" = "test-berlin" ]; then
    dir=berlin-3d-tiles
    rm -rf $dir
    mkdir $dir
    tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/triangle-berlin.gml -o ${dir} --utm_zone 33 --utm_hemisphere N -b 100 --dont_save_textures --number_of_buildings 1
elif [ "${CITY}" = "nyc" ]; then
    dir=ny-3d-tiles
    rm -rf $dir
    mkdir $dir
else
    print_parameters "$0"
    exit 1
fi
