#pragma once

#include <optional>
#include <utility>

#include <torch/torch.h>


const double EPS = std::numeric_limits<double>::epsilon();

class CouplingTransform {
public:
    explicit CouplingTransform();
    virtual ~CouplingTransform() = default;

    void invert() { inverted = !inverted; }
    bool is_inverted() const { return inverted; }

    virtual std::pair<at::Tensor, std::optional<at::Tensor>> operator()(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian
    ) const final;

protected:
    bool inverted;

    virtual std::pair<at::Tensor, std::optional<at::Tensor>> forward(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian = true
    ) const = 0;

    virtual std::pair<at::Tensor, std::optional<at::Tensor>> backward(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian = true
    ) const = 0;
};


class PWLinearCouplingTransform : public CouplingTransform {
public:
    using CouplingTransform::CouplingTransform;

protected:
    std::pair<at::Tensor, std::optional<at::Tensor>> forward(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian = true
    ) const override;

    std::pair<at::Tensor, std::optional<at::Tensor>> backward(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian = true
    ) const override;
};


class PWQuadraticCouplingTransform : public CouplingTransform {
public:
    using CouplingTransform::CouplingTransform;

protected:
    std::pair<at::Tensor, std::optional<at::Tensor>> forward(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian = true
    ) const override;

    std::pair<at::Tensor, std::optional<at::Tensor>> backward(
        at::Tensor x,
        at::Tensor theta,
        bool compute_log_jacobian = true
    ) const override;
};
