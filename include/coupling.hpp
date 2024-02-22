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

protected:
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
        int64_t n_bins,
        int64_t n_hidden,
        int64_t dim_hidden
    ) : CouplingCell(
        dim,
        std::make_shared<PWLinearCouplingTransform>(),
        mask,
        [&]{
            int64_t dim_in = mask.sum().item<int64_t>();
            int64_t dim_out = dim - dim_in;
            at::Tensor out_shape = at::tensor({dim_out, n_bins});

            return std::make_shared<DNNTrainable>(
                dim_in,
                out_shape,
                n_hidden,
                dim_hidden
            );
        }()
    ) {}
};
