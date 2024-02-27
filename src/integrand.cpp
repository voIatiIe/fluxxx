#include <iostream>
#include <cmath>

#include "integrand.hpp"


Integrand::Integrand(int dim) : dim(dim), calls(0) {}

at::Tensor Integrand::operator()(at::Tensor x) {
    calls += x.size(0);

    return callable(x);
}

void Integrand::reset() {
    calls = 0;
}


GaussIntegrand::GaussIntegrand(int dim, double mu, double sigma) : Integrand(dim), mu(mu), sigma(sigma) {}

at::Tensor GaussIntegrand::callable(at::Tensor x) {
    auto normFactor = std::pow(sigma * std::sqrt(M_PI), -dim);
    auto exponent = -at::pow((x - mu) / sigma, 2).sum(1);

    return normFactor * at::exp(exponent);
}

double GaussIntegrand::target() const {
    return std::pow(std::erf(mu / (std::sqrt(2) * sigma)), dim);
}
