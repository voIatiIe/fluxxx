#include <iostream>

#include "trainer.hpp"


void Trainer::process_loss(double loss) {
    auto alpha = 0.9;

    if (!last_loss.has_value())
        last_loss = loss;
    else
        last_loss = last_loss.value() * (1 - alpha) + loss * alpha;
}


void Trainer::process_train_batch_step(torch::Tensor x, torch::Tensor px, torch::Tensor fx) {
    if (!sample_forward && step > 2) {
        auto flat_loss_mean = fx.pow(2).mean().item<double>();
        auto flat_loss_std = std::min(fx.pow(2).std().item<double>(), flat_loss_mean / 4);

        double switch_loss_threshold = flat_loss_mean - flat_loss_std;

        if (last_loss.has_value() && (last_loss.value() < switch_loss_threshold)) {
            sample_forward = true;
            std::cout << "$Switched to forward sampling mode" << std::endl;
        }
    }
}


torch::Tensor Trainer::sample(int n_points) {
    if (flow.is_inverted()) flow.invert();

    auto x = prior->forward(n_points);
    torch::NoGradGuard no_grad_guard;
    auto xj = flow.forward(x);

    return xj.detach();
}


double Trainer::train_minibatch(
    torch::Tensor x,
    torch::Tensor px,
    torch::Tensor fx,
    torch::optim::Optimizer& optimizer
) {
    if (!flow.is_inverted())
        flow.invert();

    optimizer.zero_grad();

    auto zeros = torch::zeros({x.size(0), 1});
    auto xj = torch::cat({x, zeros}, 1);

    xj = flow.forward(xj);

    x = xj.slice(1, 0, -1);
    auto log_qx = xj.slice(1, -1).squeeze(1) + prior->log_prob(x);

    auto loss_ = loss(fx, px, log_qx);
    loss_.backward();

    optimizer.step();

    return loss_.detach().item<double>();
}


void Trainer::train_batch_step(
    torch::Tensor x,
    torch::Tensor px,
    torch::Tensor fx,
    torch::optim::Optimizer& optimizer
) {
    auto minibatch_size = (int)(minibatch_share * x.size(0));

    for (int i = 0; i < x.size(0); i += minibatch_size) {
        auto end = std::min((int)x.size(0), i + minibatch_size);

        auto loss = train_minibatch(
            x.slice(0, i, end),
            px.slice(0, i, end),
            fx.slice(0, i, end),
            optimizer
        );

        process_loss(loss);
    }
}


void Trainer::train_batch(
    torch::Tensor x,
    torch::Tensor px,
    torch::Tensor fx
) {
    auto optimizer = torch::optim::Adam(flow.parameters(), torch::optim::AdamOptions(5e-4));

    for (int epoch = 0; epoch < n_epochs; ++epoch)
        train_batch_step(x, px, fx, optimizer);

    process_train_batch_step(x, px, fx);
    step += 1;
}
