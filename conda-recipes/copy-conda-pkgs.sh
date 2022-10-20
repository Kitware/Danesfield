#!/bin/bash
# Copy conda packages to a destination directory so that we can have a
# private conda channel that does not change. From that directory
# we either upload the packages to anaconda or serve them from an http server.
# We assume we have packages come from defaults, conda forge and pytorch

# copy packages from stdin. It expects the input as it is printed by `conda list`
copyPackages()
{
    if [ $# -lt 2 ]; then
        echo "Invalid number of function parameters: $#."
        echo "Use ${FUNCNAME[0]}  'src' 'dest'";
        return;
    fi
    local src=$1
    local dest=$2
    while read name; do \
        local dirPkg=$(echo $name | tr -s " " | cut -d " " -f '1-3' | sed 's/ /-/g')
        local pkg="${dirPkg}.tar.bz2"
        if [ ! -f $src/$pkg ]; then
            if [ -d $src/$dirPkg ]; then
                echo "Archiving $pkg"
                local content=$(ls -1 $src/$dirPkg | xargs)
                tar cjf $dest/$pkg -C $src/$dirPkg $content
            else
                echo "Warning: $dirPkg does not exist"
            fi
        else
            cp $src/$pkg $dest
        fi
    done
}

# copy packages from a certain channel. Expects to be in conda environment
# where packages need to be copied from
copyChannelPackages()
{
    if [ $# -lt 3 ]; then
        echo "Invalid number of function parameters: $#."
        echo "Use ${FUNCNAME[0]} 'src' 'destBase' 'channel'";
        return;
    fi
    local src=$1
    local destBase=$2
    local channel=$3
    local dest=$destBase/$channel
    mkdir -p $dest
    dest="${dest}/linux-64"
    mkdir -p $dest
    echo "Copy packages from $channel ..."
    ${CONDA_EXE} list | \
        grep $channel | copyPackages $src $dest
}

# copy conda packages
copyCondaPackages()
{
    if [ $# -lt 1 ]; then
        echo "Invalid number of function parameters: $#."
        echo "Use ${FUNCNAME[0]}  'destBase'";
        return;
    fi
    if [ -z ${CONDA_EXE+x} ]; then
        echo "CONDA_EXE is not set. Please activate the conda environment"
        return
    fi
    local src="$(${CONDA_EXE} info | grep location | cut -d ':' -f 2 | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')/../../pkgs/"
    local destBase=$1
    local dest=$destBase/defaults
    mkdir -p $dest
    dest="${dest}/linux-64"
    mkdir -p $dest
    echo "Copy packages from defaults ..."
    ${CONDA_EXE} list | \
        grep -v pytorch | \
        grep -v conda-forge | \
        grep -v kitware-danesfield | \
        grep -v '<pip>' | \
        grep -v '^#.*' | copyPackages $src $dest

    copyChannelPackages $src $destBase "pytorch"
    copyChannelPackages $src $destBase "conda-forge"
}

if [ $# -lt 1 ]; then
    echo "Invalid number of script parameters: $#. Use $0 'destBase'";
    exit 1
fi
copyCondaPackages $1
