#include "coupling.hpp"


CouplingCell::CouplingCell(
    int64_t dim,
    std::shared_ptr<CouplingTransform> transform,
    at::Tensor mask_,
    std::shared_ptr<Trainable> trainable
) : dim(dim), transform(transform), trainable(trainable) {

    register_module("trainable", trainable);

    mask_complement = torch::logical_not(mask_);

    mask = torch::cat({mask_, torch::tensor({false}, torch::kBool)});
    mask_complement = torch::cat({mask_complement, torch::tensor({false}, torch::kBool)});
}

at::Tensor CouplingCell::forward(at::Tensor xj) {
    auto x_n = xj.index_select(1, mask.nonzero().squeeze());
    auto x_m = xj.index_select(1, mask_complement.nonzero().squeeze());
    auto log_jacobian = xj.index_select(1, at::tensor({dim})).squeeze(1);

    auto transformed = transform->operator()(x_m, trainable->forward(x_n));

    at::Tensor yj = torch::zeros_like(xj);
    yj.index_put_({torch::indexing::Slice(), mask}, x_n);
    yj.index_put_({torch::indexing::Slice(), mask_complement}, std::get<0>(transformed));
    yj.index_put_({torch::indexing::Slice(), -1}, log_jacobian + std::get<1>(transformed));

    return yj;
}
