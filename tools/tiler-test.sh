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
    echo "-k       : keeps intermediate files such as GTLF and GLB"
    echo "-g       : run only conversions from GLTF to GLB and B3DM"
}

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -k|--keep-intermediate-files)
      KEEP=0 # true
      shift
      ;;
    -g|--gltf-conversions-only)
      GLTF_ONLY=0 # true
      KEEP=0 # true
      shift
      ;;
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
    if [ ! ${GLTF_ONLY} ]; then
        rm -rf ${dir}
        mkdir ${dir}
        tiler ../../../data/CORE3D/Jacksonville/building_*building*.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -b 16
    fi
elif [ "${CITY}" = "test-jacksonville" ]; then
    dir=jacksonville-3d-tiles
    if [ ! ${GLTF_ONLY} ]; then
        rm -rf ${dir}
        mkdir ${dir}
        tiler ../../../data/CORE3D/Jacksonville/triangle.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -b 2 --dont_save_textures
    fi
elif [ "${CITY}" = "berlin" ]; then
    dir=berlin-3d-tiles
    if [ ! ${GLTF_ONLY} ]; then
        rm -rf $dir
        mkdir $dir
        tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o ${dir} --utm_zone 33 --utm_hemisphere N -b 16 --dont_save_textures --number_of_buildings 17
    fi
elif [ "${CITY}" = "test-berlin" ]; then
    dir=berlin-3d-tiles
    if [ ! ${GLTF_ONLY} ]; then
        rm -rf $dir
        mkdir $dir
        tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/triangle-berlin.gml -o ${dir} --utm_zone 33 --utm_hemisphere N -b 100 --dont_save_textures --number_of_buildings 1
    fi
elif [ "${CITY}" = "nyc" ]; then
    dir=ny-3d-tiles
    rm -rf $dir
    mkdir $dir
else
    print_parameters "$0"
    exit 1
fi

cd ${dir} || exit
# convert to glb
echo "Converting to glb..."
find . -name '*.gltf' -exec bash -c 'nodejs ~/external/gltf-pipeline/bin/gltf-pipeline.js -i ${0} -o ${0%.*}.glb' {} \;
if [ ! ${KEEP} ]; then
    echo "Deleting gltf and bin files..."
    find . -name '*.gltf' -exec rm {} \;
    find . -name '*.bin' -exec rm {} \;
fi
# convert to b3dm
echo "Converting to b3dm..."
find . -name '*.glb' -exec bash -c 'nodejs ~/external/3d-tiles-tools/tools/bin/3d-tiles-tools.js glbToB3dm ${0} ${0%.*}.b3dm' {} \;
if [ ! ${KEEP} ]; then
    echo "Deleting glb files..."
    find . -name '*.glb' -exec rm {} \;
fi
