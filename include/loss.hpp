#pragma once

#include <torch/torch.h>

using Loss = at::Tensor (*)(at::Tensor, at::Tensor, at::Tensor);

at::Tensor variance_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx);
at::Tensor dkl_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx);
