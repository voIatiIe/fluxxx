#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

#include "trainable.hpp"
#include "transform.hpp"


class CouplingCell : public torch::nn::Module {
public:
    CouplingCell(
        int64_t dim,
        std::shared_ptr<CouplingTransform> transform,
        at::Tensor mask,
        std::shared_ptr<Trainable> trainable
    );

    at::Tensor forward(at::Tensor xj);

    void invert() { transform->invert(); }
    bool is_inverted() { return transform->is_inverted(); }

private:
    int64_t dim;
    std::shared_ptr<CouplingTransform> transform;
    at::Tensor mask, mask_complement;
    std::shared_ptr<Trainable> trainable;
};


class PWLinearCouplingCell : public CouplingCell {
public:
    PWLinearCouplingCell(
        int64_t dim,
        at::Tensor mask,
        std::shared_ptr<Trainable> trainable
    ) : CouplingCell(dim, std::make_shared<PWLinearCouplingTransform>(), mask, trainable) {}
};
