#include <limits>

#include "transform.hpp"


CouplingTransform::CouplingTransform() : inverted(false) {}

std::pair<at::Tensor, at::Tensor> CouplingTransform::operator()(
    at::Tensor x,
    at::Tensor theta
) const {
    if (is_inverted())
        return backward(x, theta);
    else
        return forward(x, theta);
}


std::pair<at::Tensor, at::Tensor> PWLinearCouplingTransform::forward(
    at::Tensor x,
    at::Tensor theta
) const {
    auto n_bins = theta.size(2);
    auto N = x.size(0);
    auto x_dim = x.size(1);

    theta = n_bins * at::softmax(theta, /*dim=*/2);

    auto bin_id = at::floor(n_bins * x);
    bin_id.clamp_(/*min=*/0, /*max=*/n_bins - 1);
    bin_id = bin_id.to(at::kLong);

    x.sub_(bin_id.toType(at::kFloat) / n_bins);

    auto slope = at::gather(theta, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);
    x.mul_(slope);

    at::Tensor log_jacobian = at::log(at::prod(slope, /*dim=*/1));

    auto left_integral = at::cumsum(theta, /*dim=*/2) / n_bins;
    left_integral = at::roll(left_integral, /*shifts=*/1, /*dims=*/2);
    left_integral.index_fill_(2, at::tensor({0}, at::kLong), 0);
    left_integral = at::gather(left_integral, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);

    x.add_(left_integral);
    x.clamp_(/*min=*/EPS, /*max=*/1 - EPS);

    return {x, log_jacobian};
}


std::pair<at::Tensor, at::Tensor> PWLinearCouplingTransform::backward(
    at::Tensor x,
    at::Tensor theta
) const {
    auto n_bins = theta.size(2);
    auto N = x.size(0);
    auto x_dim = x.size(1);

    theta = n_bins * at::softmax(theta, /*dim=*/2);

    auto left_integral = at::cumsum(theta, /*dim=*/2) / n_bins;
    left_integral = at::roll(left_integral, /*shifts=*/1, /*dims=*/2);
    left_integral.index_fill_(2, at::tensor({0}, at::kLong), 0);

    auto overhead = (x.unsqueeze(-1) - left_integral).detach();
    overhead.index_put_({overhead < 0}, 2);

    auto bin_id = at::argmin(overhead, /*dim=*/2);
    bin_id.clamp_(/*min=*/0, /*max=*/n_bins - 1);

    left_integral = at::gather(left_integral, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);

    auto slope = at::gather(theta, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);

    x = bin_id.toType(at::kFloat) / n_bins + (x - left_integral) / slope;
    x.clamp_(/*min=*/EPS, /*max=*/1 - EPS);

    at::Tensor log_jacobian = -at::log(at::prod(slope, /*dim=*/1));

    return {x.detach(), log_jacobian};
}

// TODO: Implement PWQuadraticCouplingTransform
