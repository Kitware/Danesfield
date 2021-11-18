#!/bin/bash

tiler_profile()
{
    PYTHONPATH=. valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes ~/projects/VTK/build/bin/vtkpython tools/tiler.py "$@"
}

tiler_debug()
{
    PYTHONPATH=. gdb --args ~/projects/VTK/build/bin/vtkpython tools/tiler.py "$@"
}

tiler()
{
    VTK_BUILD=build-cesium3dtiles
    PYTHONPATH=. ~/projects/VTK/${VTK_BUILD}/bin/vtkpython tools/tiler.py "$@"
}

print_parameters ()
{
    echo "$0 -c[|--city] jacksonville|berlin|nyc"
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
    # tiler ../../../data/CORE3D/Jacksonville/building_15_building_22.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -t 20 -n 100
    tiler ../../../data/CORE3D/Jacksonville/building_*_building_*.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -t 20 -n 100
elif [ "${CITY}" = "jacksonville-triangle" ]; then
    dir=jacksonville-triangle
    rm -rf ${dir}
    mkdir ${dir}
    tiler ../../../data/CORE3D/Jacksonville/triangle.obj -o ${dir} --utm_zone 17 --utm_hemisphere N -t 2 --dont_save_textures --content_type 2
elif [ "${CITY}" = "berlin" ]; then
    dir=berlin-3d-tiles
    rm -rf $dir
    mkdir $dir
    tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o ${dir} --utm_zone 33 --utm_hemisphere N -t 200 --dont_save_textures --number_of_buildings 10000 --content_type 1
elif [ "${CITY}" = "berlin-stadium" ]; then
    dir=${CITY}
    rm -rf $dir
    mkdir $dir
    tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml-stadium.gml -o ${dir} --crs EPSG:25833 -t 100 --dont_save_textures --number_of_buildings 1
elif [ "${CITY}" = "berlin-stadium10" ]; then
    dir=${CITY}
    rm -rf $dir
    mkdir $dir
    tiler ../../../data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o ${dir} --crs EPSG:25833 -t 100 --dont_save_textures -b 2800 -e 3100 --content_type 1
elif [ "${CITY}" = "nyc" ]; then
    for i in {9..9}; do
        dir=nyc${i}-3d-tiles
        rm -rf $dir
        mkdir $dir
        tiler ../../../data/NYC-3D-Building/DA_WISE_GMLs/DA${i}_3D_Buildings_Merged.gml -o ${dir} --crs EPSG:2263 -t 100 --dont_save_textures -n 1 --content_type 2
    done
elif [ "${CITY}" = "nyc10" ]; then
    dir=ny-3d-tiles
    rm -rf $dir
    mkdir $dir
    tiler ../../../data/NYC-3D-Building/DA_WISE_GMLs/DA10_3D_Buildings_Merged.gml -o ${dir} --crs EPSG:2263 -t 100 --dont_save_textures -n 10000
else
    echo "Error: Cannot find ${CITY}"
    print_parameters "$0"
    exit 1
fi
