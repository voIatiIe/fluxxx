#include <torch/torch.h>

#include "lorentz.hpp"

#include <iostream>

const double FMAX = std::numeric_limits<float>::max();
const double FEPS = std::sqrt(std::numeric_limits<float>::epsilon());


at::Tensor _dot(at::Tensor x, at::Tensor y) {
    return x.select(1, 0) * y.select(1, 0) -
           x.select(1, 1) * y.select(1, 1) -
           x.select(1, 2) * y.select(1, 2) -
           x.select(1, 3) * y.select(1, 3);
}

at::Tensor _square(at::Tensor x) {
    if (x.size(1) == 4 || x.size(0) == 4)
        return _dot(x, x);

    return at::sum(x*x, -1);
}

at::Tensor _rho(at::Tensor x) {
    return at::sum(x.slice(1, 1).pow(2), -1);
}

at::Tensor beta(at::Tensor x) {
    return x.slice(1, 1) / x.select(1, 0).unsqueeze(1);
}

at::Tensor set_square(at::Tensor x, at::Tensor square) {
    auto ret = at::zeros_like(x);

    ret.select(1, 0) = at::sqrt(_rho(x) + square);    
    ret.slice(1, 1) = x.slice(1, 1);

    return ret;
}

at::Tensor boost(at::Tensor x, at::Tensor beta) {
    auto b2 = _square(beta);
    auto gamma = 1.0 / at::sqrt(1.0 - b2);

    auto spacial = x.slice(1, 1);
    auto bp = at::sum(spacial * beta, -1);

    auto gamma2 = at::where(b2 > 0, (gamma - 1.0) / b2, at::zeros_like(b2));
    auto factor = gamma2 * bp + gamma * x.select(1, 0);

    x.select(1, 0) = gamma * (x.select(1, 0) + bp);
    x.slice(1, 1) = x.slice(1, 1) + factor.unsqueeze(-1) * beta;

    return x;
}

at::Tensor pseudo_rapidity(at::Tensor x) {
    auto pt = at::sqrt(at::sum(at::pow(x.slice(1, 1, 3), 2), -1));
    auto th = at::atan2(pt, x.select(1, 3));

    auto condition = (pt < FEPS) & (at::abs(x.select(1, 3)) < FEPS);

    return at::where(
        condition,
        FMAX * at::ones_like(x.select(1, 3)),
        -at::log(at::tan(th / 2))
    );
}

at::Tensor delta_phi(at::Tensor x, at::Tensor y) {
    auto pt_x = at::sqrt(at::sum(at::pow(x.slice(1, 1, 3), 2), -1));
    auto pt_y = at::sqrt(at::sum(at::pow(y.slice(1, 1, 3), 2), -1));

    auto tmp = (x.select(1, 1) * y.select(1, 1) + x.select(1, 2) * y.select(1, 2)) / (pt_x * pt_y);

    auto dphi = at::where(
        at::abs(tmp) > at::ones_like(tmp),
        at::acos(tmp / at::abs(tmp)),
        at::acos(tmp)
    );

    dphi = at::where(
        (pt_x == 0) | (pt_y == 0),
        FMAX * at::ones_like(dphi),
        dphi
    );

    return dphi;
}

at::Tensor deltaR(at::Tensor x, at::Tensor y) {
    auto delta_eta = pseudo_rapidity(x) - pseudo_rapidity(y);
    auto dphi = delta_phi(x, y);

    return at::sqrt(delta_eta.pow(2) + dphi.pow(2));
}
