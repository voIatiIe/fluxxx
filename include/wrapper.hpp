#pragma once

#include <iostream>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <torch/torch.h>


class MatrixWrapper {
public:
    MatrixWrapper();
    ~MatrixWrapper();
    double processTensor(const std::vector<std::vector<std::vector<double>>>& tensor);

    double smatrix(torch::Tensor tensor);
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

MatrixWrapper::MatrixWrapper() {
    Py_Initialize();

    auto init_numpy = []() -> bool {
        import_array();
        return true;
    }();

    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString(modulePath.c_str()));

    PyObject* pModuleName = PyUnicode_FromString(moduleName.c_str());
    pModule = PyImport_Import(pModuleName);
    Py_DECREF(pModuleName);

    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load module " << pModuleName << std::endl;
    }

    pSmatrix = PyObject_GetAttrString(pModule, smatrixName.c_str());
    if (!pSmatrix || !PyCallable_Check(pSmatrix)) {
        PyErr_Print();
        std::cerr << "Can't find function " << smatrixName << std::endl;
    }

    pInitialiseModel = PyObject_GetAttrString(pModule, initialiseModelName.c_str());
    if (!pInitialiseModel || !PyCallable_Check(pInitialiseModel)) {
        PyErr_Print();
        std::cerr << "Can't find function " << initialiseModelName << std::endl;
    }
}

MatrixWrapper::~MatrixWrapper() {
    Py_XDECREF(pSmatrix);
    Py_XDECREF(pInitialiseModel);
    Py_XDECREF(pModule);
    Py_Finalize();
}


double MatrixWrapper::smatrix(torch::Tensor tensor) {
    auto sizes = tensor.sizes();
    npy_intp dims[sizes.size()];

    for (size_t i = 0; i < sizes.size(); ++i)
        dims[i] = static_cast<npy_intp>(sizes[i]);

    PyObject* pArray = PyArray_SimpleNewFromData(
        tensor.dim(),
        dims,
        NPY_FLOAT,
        tensor.data_ptr()
    );

    PyObject* pArgs = PyTuple_Pack(1, pArray);
    PyObject* pResult = PyObject_CallObject(pSmatrix, pArgs);

    Py_DECREF(pArgs);
    Py_DECREF(pArray);

    if (pResult != nullptr) {
        double result = PyFloat_AsDouble(pResult);
        Py_DECREF(pResult);
        return result;
    } else {
        PyErr_Print();
        return -0.0;
    }
}

void MatrixWrapper::initialisemodel() {
    PyObject* pArgs = PyTuple_New(0);

    PyObject_CallObject(pInitialiseModel, pArgs);

    Py_DECREF(pArgs);
}
