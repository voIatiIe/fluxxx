#pragma once

#include <torch/torch.h>


class Trainable : public torch::nn::Module {
public:
    Trainable(int64_t dim_in, at::Tensor out_shape);

    at::Tensor forward(at::Tensor x);

protected:
    int64_t dim_in;
    at::Tensor out_shape;
    int64_t dim_out;
    torch::nn::Sequential trainable;
};


class DNNTrainable : public Trainable {
public:
    DNNTrainable(
        int64_t dim_in,
        at::Tensor out_shape,
        int64_t n_hidden,
        int64_t dim_hidden
    );
};
