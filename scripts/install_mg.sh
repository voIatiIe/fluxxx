set -e

path=$(realpath $0)
scripts_path=$(dirname $path)
root_path=$(dirname $scripts_path)

python_packages=$root_path/python/lib/python3.8/site-packages
export LD_LIBRARY_PATH=$root_path/python/lib

if [ ! -d $root_path/mg5 ]
then
    mkdir $root_path/mg5
    wget https://launchpad.net/mg5amcnlo/lts/2.8.x/+download/MG5_aMC_v2.8.3.2.tar.gz
    tar -xzf MG5_aMC_v2.8.3.2.tar.gz -C $root_path/mg5 --strip-components=1
    rm MG5_aMC_v2.8.3.2.tar.gz

    $root_path/python/bin/python3.8 -m pip install --target=$python_packages -r $scripts_path/requirements.txt
    $root_path/python/bin/python3.8 $root_path/mg5/bin/mg5_aMC --logging=ERROR --file=$scripts_path/lhapdf.mg5

    wget http://lhapdfsets.web.cern.ch/lhapdfsets/current/NNPDF23_nlo_as_0119.tar.gz
    tar -xzf NNPDF23_nlo_as_0119.tar.gz -C $root_path/mg5/HEPTools/lhapdf6_py3/
    rm NNPDF23_nlo_as_0119.tar.gz
fi
