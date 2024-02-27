#include <iostream>
#include <tuple>

#include "integrator.hpp"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Integrator::sample_refine() {
    auto xj = trainer.sample(n_points_refine);
    auto x = xj.slice(/*dim=*/1, /*start=*/0, /*end=*/-1);

    auto px = torch::exp(-xj.slice(/*dim=*/1, /*start=*/-1).squeeze(1));
    auto fx = integrand -> operator()(x);

    return std::make_tuple(x, px, fx);
}


void Integrator::refine_step() {
    auto res = sample_refine();

    auto x = std::get<0>(res);
    auto px = std::get<1>(res);
    auto fx = std::get<2>(res);

    auto var_mean = torch::var_mean(fx / px);

    std::cout << "Refine step: " << std::get<1>(var_mean).item<double>() << " +- " << std::get<0>(var_mean).item<double>() << " Target " << integrand -> target() << std::endl;
}


void Integrator::refine() {
    std::cout << "Refine phase... " << n_iter_refine << std::endl;
    for (int _ = 0; _ < n_iter_refine; _++)
        refine_step();
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Integrator::sample_survey() {
    at::Tensor xj;

    if (trainer.sample_forward)
        xj = trainer.sample(n_points_survey);
    else
        xj = posterior.forward(n_points_survey);

    auto x = xj.slice(/*dim=*/1, /*start=*/0, /*end=*/-1);
    auto px = torch::exp(-xj.slice(/*dim=*/1, /*start=*/-1).squeeze(1));
    auto fx = integrand -> operator()(x);

    return std::make_tuple(x, px, fx);
}


void Integrator::survey_step() {
    auto res = sample_survey();

    auto x = std::get<0>(res);
    auto px = std::get<1>(res);
    auto fx = std::get<2>(res);

    auto var_mean = torch::var_mean(fx / px);

    trainer.train_batch(x, px, fx);

    std::cout << "Survey step: " << std::get<1>(var_mean).item<double>() << " +- " << std::get<0>(var_mean).item<double>() << " Target " << integrand -> target() << std::endl;
}


void Integrator::survey() {
    std::cout << "Survey phase..." << std::endl;
    for (int _ = 0; _ < n_iter_survey; _++)
        survey_step();
}


void Integrator::integrate() {
    survey();
    refine();
}
