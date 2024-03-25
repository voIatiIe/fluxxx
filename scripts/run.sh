set -e

path=$(realpath $0)
scripts_path=$(dirname $path)
root_path=$(dirname $scripts_path)

delete_processes() {
    for folder in $root_path/mg5/fortran_output/SubProcesses/P*; do
        if [ -d $folder ]; then
            echo "Deleting folder: $folder"
            rm -rf $folder
        fi
    done
}
delete_processes

export LD_LIBRARY_PATH=$dir_path/MG5_aMC_v2_8_3_2/HEPTools/lhapdf6_py3/lib
export LD_LIBRARY_PATH=$root_path/python/lib

$root_path/python/bin/python3.8 $root_path/mg5/bin/mg5_aMC --logging=ERROR --file=$scripts_path/run.mg5

cd $root_path/mg5/fortran_output/SubProcesses/P*/
make matrix2py.so
rm -f matrix2py.so
mv matrix2py* matrix2py.so
cd -

cp $root_path/mg5/fortran_output/Cards/{param_card.dat,ident_card.dat} $root_path/params
cp $root_path/mg5/fortran_output/SubProcesses/P*/{param.log,nexternal.inc,pmass.inc,matrix2py.so} $root_path/params
