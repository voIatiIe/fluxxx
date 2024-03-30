#include <cmath>

#include "integrand.hpp"
#include "wrapper.hpp"

#include <iostream>


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


MGIntegrand::MGIntegrand(
    double E,
    std::vector<double> initial_masses,
    std::vector<double> final_masses
) : Integrand(1), E(E), generator(PhaseSpaceGenerator(initial_masses, final_masses)) {
    dim = generator.n_dims();

    wrapper.initialisemodel();
}


at::Tensor MGIntegrand::callable(at::Tensor x) {
    auto res = generator.generate_kinematics_batch(E, x, 10.0, 0.4, 2.5);

    auto mrx = wrapper.smatrix(std::get<0>(res).to(at::kDouble));
    auto jac = std::get<1>(res).to(at::kDouble);

    const double NORM = 2.5681894616e-9;

    return mrx * jac / NORM;
}


double MGIntegrand::target() const {
    return 0.0;
}
