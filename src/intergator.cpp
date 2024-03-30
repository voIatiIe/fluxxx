#include <iostream>
#include <tuple>

#include "integrator.hpp"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Integrator::sample_survey() {
    at::Tensor xj;

    if (trainer.sample_forward)
        xj = trainer.sample(n_points_survey);
    else
        xj = posterior->forward(n_points_survey);

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

    auto n_pts = x.sizes()[0];

    auto mean = std::get<1>(var_mean).item<double>();
    auto unc = std::sqrt(std::get<0>(var_mean).item<double>() / n_pts);

    n_survey_step++;
    std::cout << "Survey step [" << n_survey_step << "/" << n_survey_steps << "]: " << mean << " +- " << unc << std::endl;
}


void Integrator::survey() {
    std::cout << "Survey phase..." << std::endl;
    for (int _ = 0; _ < n_survey_steps; _++)
        survey_step();
}


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

    auto n_pts = x.sizes()[0];

    auto mean = std::get<1>(var_mean).item<double>();
    auto unc = std::sqrt(std::get<0>(var_mean).item<double>() / n_pts);

    integrals.push_back(mean);
    uncertainties.push_back(unc);

    n_refine_step++;
    std::cout << "Refine step [" << n_refine_step << "/" << n_refine_steps << "]: " << mean << " +- " << unc << std::endl;
}


void Integrator::refine() {
    std::cout << "Refine phase... " << n_refine_steps << std::endl;
    for (int _ = 0; _ < n_refine_steps; _++)
        refine_step();
}


void Integrator::finalize() {
    auto n_estimates = integrals.size();

    double mean = 0;
    for (int i = 0; i < n_estimates; i++)
        mean += integrals[i];
    
    mean /= n_estimates;

    double unc = 0;
    for (int i = 0; i < n_estimates; i++)
        unc += uncertainties[i] * uncertainties[i];

    unc = std::sqrt(unc) / n_estimates;

    std::cout << "Estimate: " << mean << " +- " << unc << std::endl;
    std::cout << "Target: " << integrand -> target() << std::endl;
}


void Integrator::integrate() {
    survey();
    refine();
    finalize();
}
