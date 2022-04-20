#!/bin/bash

tiler()
{
    PYTHONPATH="${SCRIPT_DIR}/.." "${VTK_DIR}/bin/vtkpython" "${SCRIPT_DIR}/tiler.py" "$@"
}

print_parameters ()
{
    echo "$0 -c[|--city] jacksonville|berlin|nyc|rio-points|..."
    echo "-c <city>: selects the city to convert to 3D Tiles"
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
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=~/data
VTK_DIR=~/projects/VTK/build-3dtiles-pointcloud
echo "$SCRIPT_DIR" "$DATA_DIR" "$VTK_DIR"

if [ "${CITY}" = "jacksonville" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_*_building_*.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 20 -n 100)
elif [ "${CITY}" = "jacksonville-gltf" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_*_building_*.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 20 -n 100 --content_gltf)
elif [ "${CITY}" = "jacksonville-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_*_building_*.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 10000 --input_type 1)
elif [ "${CITY}" = "jacksonville-points-gltf" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_*_building_*.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 10000 --input_type 1 --content_gltf)
elif [ "${CITY}" = "jacksonville-triangle" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/triangle.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 2 --dont_save_textures --content_gltf --input_type 0)
elif [ "${CITY}" = "jacksonville-mesh" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_ground.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 10000 --dont_save_textures --content_gltf --input_type 2)
elif [ "${CITY}" = "berlin" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml -o "${CITY}" --utm_zone 33 --utm_hemisphere N -t 200 --dont_save_textures --number_of_features 10000 --input_type 0)
elif [ "${CITY}" = "berlin-stadium" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/Berlin-3D/Charlottenburg-Wilmersdorf/citygml-stadium.gml -o "${CITY}" --crs EPSG:25833 -t 100 --number_of_features 1)
elif [ "${CITY}" = "nyc" ]; then
    i=1
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}/NYC-3D-Building/DA_WISE_GMLs/DA${i}_3D_Buildings_Merged.gml" -o "${CITY}" --crs EPSG:2263 -t 100 --dont_save_textures --content_gltf -m -n 10000)
elif [ "${CITY}" = "rio-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/nga_data/RoI-keep_xy_664000_7471500-665000_7472500.las -o "${CITY}" --crs EPSG:32723 -t 10000 --input_type 1 --points_color_array Color)
elif [ "${CITY}" = "aphill-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/TeleSculptor/examples/09172008flight1tape3_2/results/textured_mesh.vtp -o "${CITY}" --utm_zone 18 --utm_hemisphere N --translation 293513 4229533 -71.6739744841 -t 10000 --input_type 1 --points_color_array mean)
else
    echo "Error: Cannot find ${CITY}"
    print_parameters "$0"
    exit 1
fi
echo "${CMD[*]}"
${CMD[*]}
