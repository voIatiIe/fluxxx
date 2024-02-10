#include "mask.hpp"
#include <cmath>


Mask::Mask(int dim) : dim(dim) {}

std::vector<std::vector<bool>> Mask::operator()(std::optional<std::size_t> n) const {
    auto masks_ = masks();

    if (!n.has_value()) n = (dim > 5) ? 2 * ceil(log2(dim)) : dim;

    std::size_t n_masks = n.value();
    std::size_t n_copies = n_masks / masks_.size();
    std::size_t n_addition = n_masks % masks_.size();

    std::vector<std::vector<bool>> extended;

    for (std::size_t i = 0; i < n_copies; ++i)
        extended.insert(extended.end(), masks_.begin(), masks_.end());

    extended.insert(extended.end(), masks_.begin(), masks_.begin() + n_addition);

    return extended;
}


std::vector<std::vector<bool>> CheckerboardMask::masks() const {
    std::vector<std::vector<bool>> _masks;

    for (std::size_t mod = 1; mod < dim; mod *= 2) {
        std::vector<bool> _mask(dim);

        for (int i = 0; i < dim; ++i) _mask[i] = (i / mod) % 2 == 0;
        _masks.push_back(_mask);

        for (int i = 0; i < dim; ++i) _mask[i] = !_mask[i];
        _masks.push_back(_mask);
    }

    return _masks;
}
