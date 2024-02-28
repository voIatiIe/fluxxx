#include <cmath>

#include "distribution.hpp"


Distribution::Distribution(int dim) : dim(dim) {}

UniformDistribution::UniformDistribution(int dim) : Distribution(dim) {}

at::Tensor UniformDistribution::sample(at::IntArrayRef size) const {
    return at::rand(size);
}

at::Tensor UniformDistribution::log_prob(at::Tensor x) const { return at::zeros_like(x); }


NormalDistribution::NormalDistribution(int dim, double mu, double sigma) : Distribution(dim), mu(mu), sigma(sigma) {}

at::Tensor NormalDistribution::sample(at::IntArrayRef size) const {
    return at::normal(mu, sigma, size);
}

at::Tensor NormalDistribution::log_prob(at::Tensor x) const {
    auto norm_factor = std::pow(sigma * std::sqrt(M_PI * 2), -1);
    auto exponent = -at::pow((x - mu) / sigma, 2) / 2;

    return at::log(at::tensor(norm_factor)) + exponent;
}
