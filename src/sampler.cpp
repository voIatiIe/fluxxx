#include <memory>

#include "sampler.hpp"


Sampler::Sampler(int dim, std::shared_ptr<Distribution> prior) : dim(dim), prior(prior) {}

at::Tensor Sampler::log_prob(at::Tensor x) const { return prior->log_prob(x); }

at::Tensor Sampler::forward(int n_points) const {
    auto sample = prior->sample({ n_points, dim });
    auto log_j = -log_prob(sample);

    return at::cat({ sample, log_j.unsqueeze(-1) }, -1);
}
