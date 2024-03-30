#include <memory>

#include "sampler.hpp"


Sampler::Sampler(int dim, std::shared_ptr<Distribution> prior) : dim(dim), prior(prior) {}

at::Tensor Sampler::log_prob(at::Tensor x) const { return prior->log_prob(x).sum(-1); }

at::Tensor Sampler::forward(int n_points) const {
    auto sample = prior->sample({ n_points, dim });
    auto log_j = -log_prob(sample);

    return at::cat({ sample, log_j.unsqueeze(-1) }, -1);
}


at::Tensor PaddedUniformSampler::forward(int n_points) const {
    const double padding = 10e-5;

    auto sample = prior->sample({ n_points, dim });
    sample.clamp_(padding, 1 - padding);

    auto log_j = -log_prob(sample);

    return at::cat({ sample, log_j.unsqueeze(-1) }, -1);
}
