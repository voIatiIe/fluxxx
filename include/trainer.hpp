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
        Sampler prior,
        Loss loss
    ): flow(flow), prior(prior), loss(loss) {};

    torch::Tensor sample(int n_points);

    void process_loss(double loss);
    void process_train_batch_step(torch::Tensor x, torch::Tensor px, torch::Tensor fx);

    torch::Tensor train_minibatch(
        torch::Tensor x,
        torch::Tensor px,
        torch::Tensor fx,
        torch::optim::Optimizer& optimizer
    );

    void train_batch_step(
        torch::Tensor x,
        torch::Tensor px,
        torch::Tensor fx,
        torch::optim::Optimizer& optimizer,
        float minibatch_share
    );

    void train_batch(
        torch::Tensor x,
        torch::Tensor px,
        torch::Tensor fx,
        int n_epochs = 20,
        float minibatch_share = 1.0
    );

private:
    Flow flow;
    Sampler prior;
    Loss loss;

    int step = 0;
    bool sample_forward = false;
    std::optional<double> last_loss = std::nullopt;
};