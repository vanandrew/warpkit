#!/bin/bash
# Installs warpkit via conda (actually micromamba)
# this creates a micromamba folder in the HOME directory by default
# but it can be changed if the user wants via the first argument

# test if curl exists on the system
if ! command -v curl &> /dev/null
then
    echo "curl could not be found"
    echo "please install curl and try running this script again"
    exit 1
fi

# check if meta.yaml is in the current directory
if [ ! -f "meta.yaml" ]
then
    echo "meta.yaml not found in current directory"
    echo "please run this script from the conda directory of the repo"
    exit 1
fi

# create a parser for the arguments with help text
# the first argument should be path to a new micromamba folder
# but if not specified, just use this directory
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -p|--path)
    MICROMAMBA_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    echo "Usage: warpkit-conda-install.sh [-p|--path <path_to_micromamba_folder>] [-h|--help]"
    exit 0
    ;;
    *)    # unknown option
    echo "Unknown option: $1"
    echo "Usage: warpkit-conda-install.sh [-p|--path <path_to_micromamba_folder>] [-h|--help]"
    exit 1
    ;;
esac
done
if [ -z "$MICROMAMBA_PATH" ]
then
    MICROMAMBA_PATH="${HOME}/micromamba"
fi

# always delete and recreate the micromamba folder
# but if the folder exists, ask the user if they want to delete it
if [ -d "$MICROMAMBA_PATH" ]; then
    echo "The path at $(realpath ${MICROMAMBA_PATH}) already exists."
    read -p "Do you want to delete it and its contents? (yes/[no]) " answer
    if [ "$answer" = "yes" ]; then
        rm -rf "$MICROMAMBA_PATH"
    else
        echo "Aborting installation."
        exit 1
    fi
fi
mkdir -p "$MICROMAMBA_PATH"

# get the aboslute path to the micromamba folder
MICROMAMBA_PATH=$(realpath "$MICROMAMBA_PATH")
echo "Using micromamba path: $MICROMAMBA_PATH"

# download micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MICROMAMBA_PATH" bin/micromamba

# now set the micromamba prefix to the MICROAMBA_PATH
export MAMBA_ROOT_PREFIX="${MICROMAMBA_PATH}"
eval "$("${MICROMAMBA_PATH}"/bin/micromamba shell hook -s posix)"

# activate the base environment
micromamba activate base

# configure the environment to use conda-forge
echo "Appending conda-forge to micromamba config..."
micromamba config append channels conda-forge
echo "Removing defaults from micromamba config..."
micromamba config remove channels defaults || true  # not an error if defaults not present

# install boa
micromamba install --yes boa || exit 1

# build warpkit conda package
conda mambabuild . || exit 1

# install warpkit conda package
micromamba install --yes -c local warpkit || exit 1

# Yell at user about setting the environment variable and eval hook
clear
echo "warpkit should now be installed!"
echo ""
echo "IMPORTANT: Please add the following lines to your .bashrc or .zshrc file:"
echo ""
echo export MAMBA_ROOT_PREFIX="${MICROMAMBA_PATH}"  # optional, defaults to ~/micromamba
echo eval \"\$\("${MICROMAMBA_PATH}"/bin/micromamba shell hook \-s posix\)\"
echo ""
echo "Then to get back into the conda environment for warpkit, run:"
echo ""
echo "micromamba activate base"
