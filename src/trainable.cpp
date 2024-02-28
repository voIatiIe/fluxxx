#include "trainable.hpp"


Trainable::Trainable(int64_t dim_in, at::Tensor out_shape) : dim_in(dim_in), out_shape(out_shape) {
    dim_out = out_shape.prod().item<int64_t>();
}

at::Tensor Trainable::forward(at::Tensor x) {
    std::vector<int> out_shape_vector;
    std::vector<int64_t> view_sizes = {x.sizes()[0]};

    for (int i = 0; i < out_shape.sizes()[0]; ++i)
        out_shape_vector.push_back(out_shape[i].item<int>());

    view_sizes.insert(view_sizes.end(), out_shape_vector.begin(), out_shape_vector.end());

    return trainable->forward(x).view(view_sizes);
}


DNNTrainable::DNNTrainable(
    int64_t dim_in,
    at::Tensor out_shape,
    int64_t n_hidden,
    int64_t dim_hidden
) : Trainable(dim_in, out_shape) {

    trainable->push_back(torch::nn::Linear(dim_in, dim_hidden));
    trainable->push_back(torch::nn::ReLU());

    for (int64_t i = 0; i < n_hidden; ++i) {
        trainable->push_back(torch::nn::Linear(dim_hidden, dim_hidden));
        trainable->push_back(torch::nn::ReLU());
    }

    trainable->push_back(torch::nn::Linear(dim_hidden, dim_out));

    register_module("inner_trainable", trainable);
}
