#include <torch/torch.h>


at::Tensor _dot(at::Tensor x, at::Tensor y);
at::Tensor _square(at::Tensor x);
at::Tensor _rho(at::Tensor x);
at::Tensor beta(at::Tensor x);
at::Tensor set_square(at::Tensor x, at::Tensor square);
at::Tensor boost(at::Tensor x, at::Tensor beta);
at::Tensor pseudo_rapidity(at::Tensor x);
at::Tensor delta_phi(at::Tensor x, at::Tensor y);
at::Tensor deltaR(at::Tensor x, at::Tensor y);
