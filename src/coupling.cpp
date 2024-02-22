#include "coupling.hpp"


CouplingCell::CouplingCell(
    int64_t dim,
    std::shared_ptr<CouplingTransform> transform,
    at::Tensor mask,
    std::shared_ptr<Trainable> trainable
) : dim(dim), transform(transform), mask(mask), trainable(trainable) {

    mask_complement = torch::logical_not(mask);

    mask = torch::cat({mask, torch::tensor({false}, torch::kBool)});
    mask_complement = torch::cat({mask_complement, torch::tensor({false}, torch::kBool)});
}

at::Tensor CouplingCell::forward(at::Tensor xj) {
    auto x_n = xj.index({torch::indexing::Slice(), mask});
    auto x_m = xj.index({torch::indexing::Slice(), mask_complement});

    auto log_jacobian = xj.index({torch::indexing::Slice(), -1});

    at::Tensor yj = torch::zeros_like(xj);

    yj.index_put_({torch::indexing::Slice(), mask}, x_n);

    auto transformed = transform->operator()(x_m, trainable->forward(x_n));

    yj.index_put_({torch::indexing::Slice(), mask_complement}, std::get<0>(transformed));
    yj.index_put_({torch::indexing::Slice(), -1}, log_jacobian + std::get<1>(transformed));

    return yj;
}
