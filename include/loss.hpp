#pragma once

#include <torch/torch.h>


at::Tensor variance_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx);
at::Tensor dkl_loss(at::Tensor fx, at::Tensor px, at::Tensor log_qx);
