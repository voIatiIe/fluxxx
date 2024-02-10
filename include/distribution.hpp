#pragma once

#include <torch/torch.h>


class Distribution {
public:
    explicit Distribution(int dim);
    virtual ~Distribution() = default;

    virtual at::Tensor sample(at::IntArrayRef size) const = 0;
    virtual at::Tensor log_prob(at::Tensor x) const = 0;

protected:
    int dim;
};


class UniformDistribution : public Distribution {
public:
    UniformDistribution(int dim);

    at::Tensor sample(at::IntArrayRef size) const override;
    at::Tensor log_prob(at::Tensor x) const override;
};


class NormalDistribution : public Distribution {
public:
    NormalDistribution(int dim, double mu = 0.0, double sigma = 1.0);

    at::Tensor sample(at::IntArrayRef size) const override;
    at::Tensor log_prob(at::Tensor x) const override;

protected:
    double mu;
    double sigma;
};
