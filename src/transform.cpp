#include <limits>

#include "transform.hpp"


CouplingTransform::CouplingTransform() : inverted(false) {}

std::pair<at::Tensor, std::optional<at::Tensor>> CouplingTransform::operator()(
    at::Tensor x,
    at::Tensor theta,
    bool compute_log_jacobian
) const {
    if (is_inverted())
        return backward(x, theta, compute_log_jacobian);
    else
        return forward(x, theta, compute_log_jacobian);
}

// TODO: Rewiew this

std::pair<at::Tensor, std::optional<at::Tensor>> PWLinearCouplingTransform::forward(
    at::Tensor x,
    at::Tensor theta,
    bool compute_log_jacobian
) const {
    auto N_ = theta.size(0);
    auto x_dim_ = theta.size(1);
    auto n_bins = theta.size(2);
    auto N = x.size(0);
    auto x_dim = x.size(1);

    TORCH_CHECK(N == N_, "Shape mismatch");
    TORCH_CHECK(x_dim == x_dim_, "Shape mismatch");

    auto bin_id = at::floor(n_bins * x);
    bin_id = at::clamp(bin_id, /*min=*/0, /*max=*/n_bins - 1);
    bin_id = bin_id.to(at::kLong);

    if (at::any(at::isnan(bin_id)).item<bool>())
        throw std::runtime_error("NaN found!");
    if (at::any(bin_id < 0).item<bool>() || at::any(bin_id >= n_bins).item<bool>())
        throw std::runtime_error("Indexing error!");

    x = x - bin_id.toType(torch::kDouble) / n_bins;
    auto slope = at::gather(theta, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);

    x *= slope;

    at::optional<at::Tensor> log_jacobian = at::nullopt;
    if (compute_log_jacobian) {
        log_jacobian = at::log(at::prod(slope, /*dim=*/1));
    }

    auto left_integral = at::cumsum(theta, /*dim=*/2) / n_bins;
    left_integral = at::roll(left_integral, /*shifts=*/1, /*dims=*/2);
    
    auto index = torch::tensor({0}, torch::kLong);
    left_integral.index_fill_(2, index, 0);

    left_integral = at::gather(left_integral, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);

    x += left_integral;

    x = at::clamp(x, /*min=*/EPS, /*max=*/1 - EPS);

    return {x, log_jacobian};
}


std::pair<at::Tensor, std::optional<at::Tensor>> PWLinearCouplingTransform::backward(
    at::Tensor x,
    at::Tensor theta,
    bool compute_log_jacobian
) const {
    auto N_ = theta.size(0);
    auto x_dim_ = theta.size(1);
    auto n_bins = theta.size(2);
    auto N = x.size(0);
    auto x_dim = x.size(1);

    TORCH_CHECK(N == N_, "Shape mismatch");
    TORCH_CHECK(x_dim == x_dim_, "Shape mismatch");

    theta = n_bins * torch::softmax(theta, /*dim=*/2);

    auto left_integral = torch::cumsum(theta, /*dim=*/2) / n_bins;
    left_integral = torch::roll(left_integral, /*shifts=*/1, /*dims=*/2);

    auto index = torch::tensor({0}, torch::kLong);
    left_integral.index_fill_(2, index, 0);

    auto overhead = (x.unsqueeze(-1) - left_integral).detach();
    overhead.index_put_({overhead < 0}, 2);

    auto bin_id = torch::argmin(overhead, /*dim=*/2);
    bin_id = torch::clamp(bin_id, /*min=*/0, /*max=*/n_bins - 1);

    if (torch::any(torch::isnan(bin_id)).item<bool>()) {
        throw std::runtime_error("NaN found!");
    }
    if (torch::any(bin_id < 0).item<bool>() || torch::any(bin_id >= n_bins).item<bool>()) {
        throw std::runtime_error("Indexing error!");
    }

    left_integral = torch::gather(left_integral, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);
    auto slope = torch::gather(theta, /*dim=*/2, /*index=*/bin_id.unsqueeze(-1)).squeeze(-1);

    auto x_ = bin_id.toType(torch::kDouble) / n_bins + (x - left_integral) / slope;

    x_ = torch::clamp(x_, /*min=*/EPS, /*max=*/1 - EPS);

    torch::optional<torch::Tensor> log_jacobian = torch::nullopt;
    if (compute_log_jacobian) {
        log_jacobian = -torch::log(torch::prod(slope, /*dim=*/1));
    }

    return {x_.detach(), log_jacobian};
}

// TODO: Implement PWQuadraticCouplingTransform
