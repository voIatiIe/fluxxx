#include <memory>
#include <torch/torch.h>

#include "integrand.hpp"
#include "sampler.hpp"
#include "mask.hpp"
#include "flow.hpp"
#include "constant.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "integrator.hpp"
#include "generator.hpp"
#include "distribution.hpp"
#include "parser.hpp"
#include "config.hpp"


void integrate_mg();


int main() {
    integrate_mg();

    return 0;
}


void integrate_mg() {
    Config config("config.cfg");

    double E = config.get<double>("E");

    torch::set_num_threads(config.get<int>("n_threads"));

    ConfigParser parser;

    auto particle_masses = parser.parse_masses();

    std::vector<double> initial_masses = particle_masses.first;
    std::vector<double> final_masses = particle_masses.second;

    MGIntegrand integrand(
        E,
        initial_masses,
        final_masses,
        /*pT_mincut=*/config.get<double>("pT_mincut"),
        /*delR_mincut=*/config.get<double>("delR_mincut"),
        /*rap_maxcut=*/config.get<double>("rap_maxcut")
    );
    int dim = integrand.dim;

    PaddedUniformSampler sampler(dim);
    Flow flow(
        dim,
        CheckerboardMask(dim)(),
        CellType::PWLINEAR,
        /*n_bins=*/config.get<int>("n_bins"),
        /*n_hidden=*/config.get<int>("n_hidden"),
        /*dim_hidden=*/config.get<int>("dim_hidden")
    );

    Trainer trainer(
        flow,
        std::make_shared<PaddedUniformSampler>(sampler),
        variance_loss,
        /*n_epochs=*/config.get<int>("n_epochs"),
        /*minibatch_share=*/config.get<double>("minibatch_share")
    );

    Integrator integrator(
        std::make_shared<MGIntegrand>(integrand),
        trainer,
        std::make_shared<PaddedUniformSampler>(sampler),
        /*n_survey_steps=*/config.get<int>("n_survey_steps"),
        /*n_refine_steps=*/config.get<int>("n_refine_steps"),
        /*n_points_survey=*/config.get<int>("n_points_survey"),
        /*n_points_refine=*/config.get<int>("n_points_refine")
    );

    integrator.integrate();
}
