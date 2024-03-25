#pragma once

#include <iostream>
#include <vector>


class MatrixWrapper {
public:
    MatrixWrapper();
    ~MatrixWrapper();
    double processTensor(const std::vector<std::vector<std::vector<double>>>& tensor);

    torch::Tensor smatrix(torch::Tensor tensor);
    void initialisemodel();

private:
    std::string moduleName = "matrix_wrapper";
    std::string modulePath = ".";
    std::string smatrixName = "smatrix";
    std::string initialiseModelName = "initialisemodel";

    PyObject* pModule;
    PyObject* pSmatrix;
    PyObject* pInitialiseModel;
};
