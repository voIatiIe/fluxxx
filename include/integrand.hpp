#pragma once

#include <torch/torch.h>

#include "generator.hpp"
#include "wrapper.hpp"

class Integrand {
public:
    Integrand(int dim);
    virtual ~Integrand() = default;

    at::Tensor operator()(at::Tensor x);
    void reset();

    virtual at::Tensor callable(at::Tensor x) = 0;
    virtual double target() const = 0;

    int dim;

protected:
    int64_t calls;
};


class GaussIntegrand : public Integrand {
public:
    GaussIntegrand(int dim, double mu = 0.5, double sigma = 0.1);

    at::Tensor callable(at::Tensor x) override;
    double target() const override;

private:
    double mu;
    double sigma;
};


class MGIntegrand : public Integrand {
public:
    MGIntegrand(
        double E,
        std::vector<double> initial_masses,
        std::vector<double> final_masses,
        double pT_mincut,
        double delR_mincut,
        double rap_maxcut
    );

    at::Tensor callable(at::Tensor x) override;
    double target() const override;

private:
    PhaseSpaceGenerator generator;
    double E;
    MatrixWrapper wrapper;
    double pT_mincut, delR_mincut, rap_maxcut;
};
