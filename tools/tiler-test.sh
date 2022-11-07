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
VTK_DIR=~/projects/VTK/build-3dtiles-add-glb
echo "$SCRIPT_DIR" "$DATA_DIR" "$VTK_DIR"

if [ "${CITY}" = "jacksonville" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_*_building_*.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 20 -n 100)
elif [ "${CITY}" = "toycity" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler  "${DATA_DIR}"/citygml/CityGML_2.0_Test_Dataset_2012-04-23/*.gml -o "${CITY}" --utm_zone 18 --utm_hemisphere N -t 2 --translation 486981.5 4421889 -10 --lod 3 --content_gltf --content_gltf_save_gltf)
elif [ "${CITY}" = "jacksonville-merged" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_*_building_*.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 20 -n 100 --content_gltf --content_gltf_save_gltf -m --merged_texture_width 1)
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
    CMD=(tiler "${DATA_DIR}"/CORE3D/Jacksonville/building_ground.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 10000 --content_gltf --input_type 2)
elif [ "${CITY}" = "jacksonville-property-texture" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/../tasks/3dtiles-property-texture/mesh/square.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 10000 --content_gltf --content_gltf_save_gltf --input_type 2)
elif [ "${CITY}" = "jacksonville-property-texture-buildings" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/../tasks/3dtiles-property-texture/buildings/*square.obj -o "${CITY}" --utm_zone 17 --utm_hemisphere N -t 2 --content_gltf --content_gltf_save_gltf --input_type 0)
elif [ "${CITY}" = "ucsd-limited-region-rgb" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/ucsd_limited_region/*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 5 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 -m)
elif [ "${CITY}" = "ucsd-all-region" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/ucsd_all_region/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 2 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 -m)
elif [ "${CITY}" = "ucsd-full-region-self-error" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/full_region_self_error/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 2 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 -m)
elif [ "${CITY}" = "ucsd-full-region-with-dist" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/full_region_with_dist/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 2 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 -m)
elif [ "${CITY}" = "ucsd-full-region-anchor-point" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/full_region/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 4 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 --property_texture_tiff_directory anchor_point -m)
elif [ "${CITY}" = "ucsd-full-region-ppe" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/full_region/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 4 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 --property_texture_tiff_directory ppe -m)
elif [ "${CITY}" = "ucsd-full-region-unmodeled" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/full_region/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 4 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 --property_texture_tiff_directory unmodeled -m)
elif [ "${CITY}" = "ucsd-all-total-error" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/CORE3D/UCSD_chet/ucsd_all_total_error/building_*.obj -o "${CITY}" --utm_zone 11 --utm_hemisphere N -t 2 --content_gltf --content_gltf_save_gltf --input_type 0 --property_texture_png_index 1 -m)
elif [ "${CITY}" = "berlin" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/Berlin-3D/Charlottenburg-Wilmersdorf/citygml.gml "${DATA_DIR}"/Berlin-3D/Friedrichshain-Kreuzberg/citygml.gml -b 0 -e 200 -o "${CITY}" --utm_zone 33 --utm_hemisphere N -t 20 --number_of_features 10000 --input_type 0 --content_gltf --content_gltf_save_gltf -m)
elif [ "${CITY}" = "berlin-stadium" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/Berlin-3D/Charlottenburg-Wilmersdorf/citygml-stadium.gml -o "${CITY}" --crs EPSG:25833 -t 100 --number_of_features 1)
elif [ "${CITY}" = "nyc" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}/NYC-3D-Building/DA_WISE_GMLs/DA1_3D_Buildings_Merged.gml" "${DATA_DIR}/NYC-3D-Building/DA_WISE_GMLs/DA2_3D_Buildings_Merged.gml" "${DATA_DIR}/NYC-3D-Building/DA_WISE_GMLs/DA3_3D_Buildings_Merged.gml" "${DATA_DIR}/NYC-3D-Building/DA_WISE_GMLs/DA4_3D_Buildings_Merged.gml" "${DATA_DIR}/NYC-3D-Building/DA_WISE_GMLs/DA5_3D_Buildings_Merged.gml" -o "${CITY}" --crs EPSG:2263 -t 100 --dont_save_textures --content_gltf)
elif [ "${CITY}" = "rio-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/nga_data/RoI-keep_xy_664000_7471500-665000_7472500.las -o "${CITY}" --crs EPSG:32723 -t 10000 --input_type 1 --points_color_array Color)
elif [ "${CITY}" = "aphill-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/TeleSculptor/examples/09172008flight1tape3_2/results/textured_mesh.vtp -o "${CITY}" --utm_zone 18 --utm_hemisphere N --translation 293513 4229533 -71.6739744841 -t 10000 --input_type 1 --points_color_array mean)
elif [ "${CITY}" = "aphill-points-gltf" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/TeleSculptor/examples/09172008flight1tape3_2/results/textured_mesh.vtp -o "${CITY}" --utm_zone 18 --utm_hemisphere N --translation 293513 4229533 -71.6739744841 -t 10000 --input_type 1 --points_color_array mean --content_gltf)
elif [ "${CITY}" = "berlin3" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler "${DATA_DIR}"/Berlin-3D/Charlottenburg-Wilmersdorf/citygml-three-buildings.gml -o "${CITY}" --crs EPSG:25833 -t 2)
elif [ "${CITY}" = "rapid3d-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler /media/videonas/fouo/projects/danesfield_courier/Rapid3D/adhoc4/filter-black-points.las -o "${CITY}" --utm_zone 18 --utm_hemisphere N -t 20000 --input_type 1 --points_color_array Color)
elif [ "${CITY}" = "ukraine-points" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler /run/user/1000/gvfs/afp-volume:host=bananas.local,user=dan.lipsa,volume=fouo/data_golden/NGA/ukraine/ukraine_sfm_[01234]_classified.laz -o "${CITY}" --utm_zone 37 --utm_hemisphere N -t 20000 --input_type 1 --points_color_array Color)
elif [ "${CITY}" = "rapid3d-gltf" ]; then
    rm -rf "${CITY}"
    mkdir "${CITY}"
    CMD=(tiler /media/videonas/fouo/projects/danesfield_courier/results/pc_texture_maps/Rapid3D/adhoc4noBlack/*.obj -o "${CITY}" --utm_zone 18 --utm_hemisphere N -t 20 --content_gltf)
else
    echo "Error: Cannot find ${CITY}"
    print_parameters "$0"
    exit 1
fi
echo "${CMD[*]}"
${CMD[*]}
