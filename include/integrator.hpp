#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "integrand.hpp"
#include "trainer.hpp"


class Integrator {
public:
    Integrator(
        std::shared_ptr<Integrand> integrand,
        Trainer trainer,
        std::shared_ptr<Sampler> posterior,
        int n_survey_steps,
        int n_refine_steps,
        int n_points_survey,
        int n_points_refine
    ):
        integrand(integrand), trainer(trainer), posterior(posterior),
        n_survey_steps(n_survey_steps), n_refine_steps(n_refine_steps), n_points_survey(n_points_survey), n_points_refine(n_points_refine) {};

    void integrate();

protected:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_refine();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_survey();

    void refine_step();
    void refine();

    void survey_step();
    void survey();

    void finalize();

private:
    std::shared_ptr<Integrand> integrand;
    Trainer trainer;
    std::shared_ptr<Sampler> posterior;
    int n_survey_steps, n_refine_steps, n_points_survey, n_points_refine;
    int n_survey_step = 0, n_refine_step = 0;

    std::vector<double> integrals, uncertainties;
};
