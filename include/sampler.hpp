#pragma once

#include <memory>
#include <torch/torch.h>

#include <distribution.hpp>


class Sampler : public torch::nn::Module {
public:
    Sampler(int dim, std::shared_ptr<Distribution> prior);
    virtual ~Sampler() = default;

    virtual at::Tensor log_prob(at::Tensor x) const;
    virtual at::Tensor forward(int n_points) const;

protected:
    int dim;
    std::shared_ptr<Distribution> prior;
};

class UniformSampler : public Sampler {
public:
    UniformSampler(int dim) : Sampler(dim, std::make_shared<UniformDistribution>(dim)) {};
};


class PaddedUniformSampler : public UniformSampler {
public:
    PaddedUniformSampler(int dim) : UniformSampler(dim) {};

    at::Tensor forward(int n_points) const;
};
