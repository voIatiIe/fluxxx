#include "loss.hpp"


at::Tensor variance_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx) {
    return at::mean(at::pow(fx, 2) / (px * at::exp(log_qx)));
}


at::Tensor dkl_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx) {
    return -at::mean(fx * log_qx / px);
}
