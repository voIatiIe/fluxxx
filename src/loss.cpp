#include <iostream>

#include "loss.hpp"


at::Tensor variance_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx) {
    return (at::pow(fx, 2) / (px * at::exp(log_qx))).mean();
}


at::Tensor dkl_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx) {
    return -(fx * log_qx / px).mean();
}
