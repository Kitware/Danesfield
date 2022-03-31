#!/bin/bash

tiler()
{
    VTK_BUILD=build-3dtiles-pointcloud
    PYTHONPATH=. ~/projects/VTK/${VTK_BUILD}/bin/vtkpython ./tools/tiler.py "$@"
}

print_parameters ()
{
    echo "$0 -c[|--city] jacksonville|berlin|nyc|rio-points"
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
    -*=) # unsupported flags
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
cd ~/projects/danesfield/danesfield/3dtiles-pointcloud || exit
DATA_DIR=../../../..

if [ "${CITY}" = "jacksonville" ]; then
    dir=jacksonville-3d-tiles
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/CORE3D/Jacksonville/building_*_building_*.obj -o "${dir}" --utm_zone 17 --utm_hemisphere N -t 20 -n 100)
elif [ "${CITY}" = "jacksonville-points" ]; then
    dir=jacksonville-3dtiles-points
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/CORE3D/Jacksonville/building_*_building_*.obj -o "${dir}" --utm_zone 17 --utm_hemisphere N -t 10000 --input_type 1)
elif [ "${CITY}" = "jacksonville-triangle" ]; then
    dir=jacksonville-triangle
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/CORE3D/Jacksonville/triangle.obj -o "${dir}" --utm_zone 17 --utm_hemisphere N -t 2 --dont_save_textures --buildings_content_type 2 --input_type 0)
elif [ "${CITY}" = "berlin" ]; then
    dir=berlin-3d-tiles
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o "${dir}" --utm_zone 33 --utm_hemisphere N -t 200 --dont_save_textures --number_of_buildings 10000 --content_type 0)
elif [ "${CITY}" = "berlin-stadium" ]; then
    dir=${CITY}
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml-stadium.gml -o "${dir}" --crs EPSG:25833 -t 100 --dont_save_textures --number_of_buildings 1)
elif [ "${CITY}" = "berlin-stadium10" ]; then
    dir=${CITY}
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o "${dir}" --crs EPSG:25833 -t 100 --dont_save_textures -b 2800 -e 3100 --content_type 1)
elif [ "${CITY}" = "nyc" ]; then
    i=1
    dir=nyc-3d-tiles
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/NYC-3D-Building/DA_WISE_GMLs/DA${i}_3D_Buildings_Merged.gml -o "${dir}" --crs EPSG:2263 -t 100 --dont_save_textures --content_type 2 -m -n 10000)
elif [ "${CITY}" = "nyc-one" ]; then
    dir=${CITY}
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/NYC-3D-Building/DA_WISE_GMLs/DA10_3D_Buildings_Merged.gml -o "${dir}" --crs EPSG:2263 -t 100 --dont_save_textures -n 1 --content_type 2)
elif [ "${CITY}" = "rio-points" ]; then
    dir=${CITY}
    rm -rf "${dir}"
    mkdir "${dir}"
    CMD=(tiler "${DATA_DIR}"/data/nga_data/RoI-keep_xy_664000_7471500-665000_7472500.las -o "${dir}" --crs EPSG:32723 -t 10000 --input_type 1 --points_color_array Color)
else
    echo "Error: Cannot find ${CITY}"
    print_parameters "$0"
    exit 1
fi
echo "${CMD[*]}"
${CMD[*]}

