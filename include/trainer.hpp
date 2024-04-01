#pragma once

#include <optional>
#include <torch/torch.h>

#include "flow.hpp"
#include "sampler.hpp"
#include "loss.hpp"


class Trainer {
public:
    Trainer(
        Flow flow,
        std::shared_ptr<Sampler> prior,
        Loss loss,
        int n_epochs,
        float minibatch_share
    ): flow(flow), prior(prior), loss(loss), n_epochs(n_epochs), minibatch_share(minibatch_share) {
        TORCH_CHECK(minibatch_share > 0.0 && minibatch_share <= 1.0, "Invalid minibatch share: ", minibatch_share);
    };

    torch::Tensor sample(int n_points);

    void process_loss(double loss);
    void process_train_batch_step(torch::Tensor x, torch::Tensor px, torch::Tensor fx);

    double train_minibatch(
        torch::Tensor x,
        torch::Tensor px,
        torch::Tensor fx,
        torch::optim::Optimizer& optimizer
    );

    void train_batch_step(
        torch::Tensor x,
        torch::Tensor px,
        torch::Tensor fx,
        torch::optim::Optimizer& optimizer
    );

    void train_batch(
        torch::Tensor x,
        torch::Tensor px,
        torch::Tensor fx
    );

    bool sample_forward = false;

public:
    Flow flow;
    std::shared_ptr<Sampler> prior;
    Loss loss;

    int step = 0;
    int n_epochs;
    float minibatch_share;
    std::optional<double> last_loss = std::nullopt;
};
