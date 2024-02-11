#pragma once

#include <memory>
#include <optional>

#include <torch/torch.h>


class Trainable : public torch::nn::Module {
public:
    Trainable(int64_t dim_in, torch::Tensor out_shape);

    torch::Tensor forward(torch::Tensor x);

protected:
    int64_t dim_in;
    torch::Tensor out_shape;
    int64_t dim_out;
    torch::nn::Sequential trainable;
};


class DNNTrainable : public Trainable {
public:
    DNNTrainable(
        int64_t dim_in,
        torch::Tensor out_shape,
        int64_t n_hidden,
        int64_t dim_hidden
    );
};
