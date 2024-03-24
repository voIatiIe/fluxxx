set -e

path=$(realpath $0)
scripts_path=$(dirname $path)
root_path=$(dirname $scripts_path)

if [ ! -d $root_path/python ]
then
    mkdir $root_path/python_
    wget https://www.python.org/ftp/python/3.8.11/Python-3.8.11.tgz
    tar -xzf Python-3.8.11.tgz -C $root_path/python_ --strip-components=1
    rm Python-3.8.11.tgz

    mkdir $root_path/python_/install
    cd $root_path/python_/install
    $root_path/python_/configure --prefix=$root_path/python --enable-shared

    mkdir $root_path/python
    sudo make install

    cd -

    sudo rm -rf $root_path/python_
fi