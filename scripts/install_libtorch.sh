set -e

path=$(realpath $0)
scripts_path=$(dirname $path)
root_path=$(dirname $scripts_path)

if [ ! -d $root_path/libtorch ]
then
    curl -sL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip -o libtorch.zip
    unzip libtorch.zip
    rm libtorch.zip
fi
