#pragma once

#include <torch/torch.h>


class Integrand {
public:
    explicit Integrand(int dim);
    virtual ~Integrand() = default;

    at::Tensor operator()(at::Tensor x);
    void reset();

    virtual at::Tensor callable(at::Tensor x) = 0;
    virtual double target() const = 0;

protected:
    int dim;
    int64_t calls;
};


class GaussIntegrand : public Integrand {
public:
    explicit GaussIntegrand(int dim, double mu = 0.5, double sigma = 0.1);

    at::Tensor callable(at::Tensor x) override;
    double target() const override;

private:
    double mu;
    double sigma;
};
