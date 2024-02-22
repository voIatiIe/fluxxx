#include <cmath>
#include <torch/torch.h>

#include "mask.hpp"


Mask::Mask(int dim) : dim(dim) {}

std::vector<at::Tensor> Mask::operator()(std::optional<std::size_t> n) const {
    auto masks_ = masks();

    if (!n.has_value()) n = (dim > 5) ? 2 * ceil(log2(dim)) : dim;

    std::size_t n_masks = n.value();
    std::size_t n_copies = n_masks / masks_.size();
    std::size_t n_addition = n_masks % masks_.size();

    std::vector<at::Tensor> extended;

    for (std::size_t i = 0; i < n_copies; ++i)
        extended.insert(extended.end(), masks_.begin(), masks_.end());

    extended.insert(extended.end(), masks_.begin(), masks_.begin() + n_addition);

    return extended;
}


std::vector<at::Tensor> CheckerboardMask::masks() const {
    std::vector<at::Tensor> _masks;

    auto indices = at::arange(0, dim, 1).to(at::kInt);

    for (int mod = 1; mod < dim; mod *= 2) {
        auto mask_tensor = (indices.div(mod, "trunc") % 2).to(at::kBool);

        _masks.push_back(mask_tensor);
        _masks.push_back(~mask_tensor);
    }

    return _masks;
}
