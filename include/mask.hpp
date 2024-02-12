#pragma once

#include <vector>
#include <optional>
#include <cstddef>

#include <torch/torch.h>


class Mask {
public:
    Mask(int dim);
    virtual ~Mask() = default;

    virtual std::vector<at::Tensor> operator()(std::optional<std::size_t> n = std::nullopt) const final;

protected:
    int dim;

    virtual std::vector<at::Tensor> masks() const = 0;
};


class CheckerboardMask : public Mask {
public:
    using Mask::Mask;

protected:
    std::vector<at::Tensor> masks() const override;
};
