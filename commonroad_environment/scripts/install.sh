#!/usr/bin/env bash

# variables
PYTHON_VERSION=3.7
JOBS=$(($(nproc)<64 ? $(nproc) : 64))
echo $JOBS
NO_ROOT="FALSE"
ENV=""

INSTALL_FOLDER="external"
INSTALL_DIR="$(pwd)/$INSTALL_FOLDER"

# Parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  -h | --help)
    echo -e "${USAGE}"
    exit 1
    ;;

  -e | --env)
    ENVIRONMENT="$2"
    ENVIRONMENT=${ENVIRONMENT%/}
    shift # past argument
    shift # past value
    ;;

  --no-root)
    NO_ROOT="TRUE"
    shift # past argument
    ;;

  --python)
    PYTHON_VERSION="$2"
    shift # past argument
    shift # past value
    ;;
  esac
done

# ---- start functions ----
function safe_cd() {
  print_info "cd to ${@}"
  cd "${@}" || exit 255
}

function print_info() {
  local BLUE="\033[0;34m"
  local NC="\033[0m"
  echo -e "${BLUE}${1}${NC}" 
}

function print_warning() {
  local RED="\033[0;31m"
  local NC="\033[0m"
  echo -e "${RED}${1}${NC}"
}

function exit_with_error() {
  print_warning "error: ${1}"
  exit 255
}

function extract() {
  FILE="${1}"
  TYPE="${2}"
  if [ "$TYPE" = "zip" ]; then
    unzip -q "$FILE"
  elif [ "$TYPE" = "tar.gz" ]; then
    tar -xf "$FILE"
  else
    exit_with_error "Unsupported archive format"
  fi
}

function get_ifnexist() {
  LINK=$2
  FILE="${LINK##*/}"
  TYPE="$1"
  wget -nv -nc "$LINK"
  extract "$FILE" "$TYPE"
}

function require_sudo() {
  if [[ $EUID -ne 0 ]]; then
    print_info "Permission required, using root."
    sudo ${@}
  else
    ${@}
  fi
}

# function to cd ./install
#BASEDIR="$(pwd)" # basedir == folder ../install
function cd_to_installdir() {
  safe_cd "${INSTALL_DIR}" 
}
# ---- end functions ----

# ---- start script ----

# activate conda env
if [[ "$ENVIRONMENT" != "" ]]; then
  # check if env exists
  print_info "start to activate conda env: $ENVIRONMENT" 
  # activate conda in bash, so that conda activate <env> works
  conda init bash 
  #eval "$(conda shell.bash hook)" #working only on conda versions 4.6+
  ENVS=$(conda env list | awk '{print $ENVIRONMENT}' )
  if [[ $ENVS = *"$ENVIRONMENT"* ]]; then
    # activate conda env
    source activate $ENVIRONMENT
    print_info "activated conda env: $(which python)" 
  else 
    # conda $ENVIRONMENT desired, but no exsisting
    exit_with_error "the conda $ENVIRONMENT failed to activte. check if you set up $ENVIRONMENT "
  fi
else
  # conda env not desired
  print_warning "not using virtual env for installation:  $(which python)" 
fi 


# Install mpi4py with conda to deal with the OpenMPI error on GLaDOS server
print_info "Installing MPI, try to remove existing installations and install with conda mpi4py"
pip uninstall --quiet -y mpi4py 
conda uninstall --quiet -y  mpi4py 
conda install --quiet -y  mpi4py || exit_with_error "conda install mpi4py failed"

print_info "When dependencies are already installed, press ctrl+c once to skip when asked for sudo password"
print_info "Installing mpi dependencies"
require_sudo apt-get install -y libopenmpi-dev  > /dev/null

print_info "Installing build dependencies"
require_sudo apt-get install -y build-essential unzip cmake  > /dev/null

print_info "Installing commonroad-driveability-checker dependencies"
require_sudo apt-get install -y libboost-dev libboost-thread-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev libcgal-dev xutils-dev libfcl-dev > /dev/null
require_sudo apt-get install -y libomp-dev libcgal-dev libgmp-dev libglu1-mesa-dev > /dev/null

# ompi_info
print_info "Installing commonroad_rl with pip"
pip install --quiet -e .\[utils_run,tests\]

cd_to_installdir 

safe_cd commonroad-drivability-checker
print_info "Building commonroad-drivability-checker"
# rebuilding cannot be avoided since this also installs packages in global scope that cannot be cached
if [ "${NO_ROOT}" == "TRUE" ]; then
  bash build.sh -e "$CONDA_PREFIX" -v $PYTHON_VERSION --serializer --install --wheel --no-root -j $JOBS > /dev/null 2>&1
else
  bash build.sh -e "$CONDA_PREFIX" -v $PYTHON_VERSION --serializer --install --wheel -j $JOBS > /dev/null 2>&1
fi

cd_to_installdir
print_info "Installation script completed, please run the unit tests to verify."
sleep 3
exit 0
