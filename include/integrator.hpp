#pragma once

#include <memory>
#include <tuple>

#include "integrand.hpp"
#include "trainer.hpp"


class Integrator {
public:
    Integrator(
        std::shared_ptr<Integrand> integrand,
        Trainer trainer,
        Sampler posterior,
        int n_iter_survey,
        int n_iter_refine,
        int n_points_survey,
        int n_points_refine
    ):
        integrand(integrand), trainer(trainer), posterior(posterior),
        n_iter_survey(n_iter_survey), n_iter_refine(n_iter_refine), n_points_survey(n_points_survey), n_points_refine(n_points_refine) {};

    void integrate();

protected:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_refine();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_survey();

    void refine_step();
    void refine();

    void survey_step();
    void survey();

private:
    std::shared_ptr<Integrand> integrand;
    Trainer trainer;
    Sampler posterior;
    int n_iter_survey, n_iter_refine, n_points_survey, n_points_refine;
};
