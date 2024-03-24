set -e

path=$(realpath $0)
scripts_path=$(dirname $path)
root_path=$(dirname $scripts_path)

export LD_LIBRARY_PATH=$dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib

$root_path/python/bin/python3.8 $root_path/mg5/bin/mg5_aMC --logging=ERROR --file=$scripts_path/run.mg5

cd $root_path/mg5/fortran_output/SubProcesses/P1_ddx_ddx_no_ag
make matrix2py.so
rm -f matrix2py.so
mv matrix2py* matrix2py.so
cd -
