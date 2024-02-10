#pragma once

#include <vector>
#include <optional>
#include <cstddef> 


class Mask {
public:
    explicit Mask(int dim);
    virtual ~Mask() = default;

    virtual std::vector<std::vector<bool>> operator()(std::optional<std::size_t> n = std::nullopt) const final;

protected:
    int dim;

    virtual std::vector<std::vector<bool>> masks() const = 0;
};


class CheckerboardMask : public Mask {
public:
    using Mask::Mask;

protected:
    std::vector<std::vector<bool>> masks() const override;
};
