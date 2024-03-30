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

void integrate();
void phasgen();
void calculate_matrix_element();
void integrate_mg(double E);


int main() {
    integrate_mg(1000.0);

    return 0;
}


void integrate_mg(double E) {
    torch::set_num_threads(8);

    ConfigParser parser;

    auto particle_masses = parser.parse_masses();

    std::vector<double> initial_masses = particle_masses.first;
    std::vector<double> final_masses = particle_masses.second;

    MGIntegrand integrand(E, initial_masses, final_masses);
    int dim = integrand.dim;

    PaddedUniformSampler sampler(dim);
    Flow flow(dim, CheckerboardMask(dim)(), CellType::PWLINEAR);

    Trainer trainer(
        flow,
        std::make_shared<PaddedUniformSampler>(sampler),
        variance_loss,
        /*n_epochs=*/10,
        /*minibatch_share=*/1.0
    );

    Integrator integrator(
        std::make_shared<MGIntegrand>(integrand),
        trainer,
        std::make_shared<PaddedUniformSampler>(sampler),
        /*n_survey_steps=*/10,
        /*n_refine_steps=*/10,
        /*n_points_survey=*/10000,
        /*n_points_refine=*/10000
    );

    integrator.integrate();
}


void calculate_matrix_element() {
    double E = 1000.0;
    std::vector<double> initial_masses = {0.0, 0.0};
    std::vector<double> final_masses = {0.0, 0.0};

    MGIntegrand integrand(E, initial_masses, final_masses);
    UniformDistribution distribution(integrand.dim);

    auto sample = distribution.sample({ 10, integrand.dim });
    auto res = integrand(sample);

    std::cout << res << std::endl;
}


void phasgen() {
    std::vector<double> initial_masses = {0.0, 0.0};
    std::vector<double> final_masses = {0.0, 0.0};

    PhaseSpaceGenerator generator(initial_masses, final_masses);

    auto result = generator.generate_kinematics_batch(1000.0, torch::tensor({{0.00001, 0.00001}}));

    std::cout << std::get<0>(result) << std::endl;
    std::cout << std::get<1>(result) << std::endl;
}


void integrate() {
    torch::set_num_threads(8);

    const int dim = 4;

    GaussIntegrand integrand(dim);
    CheckerboardMask mask(dim);
    UniformSampler sampler(dim);

    Flow flow(dim, mask(), CellType::PWLINEAR);

    Trainer trainer(
        flow,
        std::make_shared<UniformSampler>(sampler),
        variance_loss,
        /*n_epochs=*/10,
        /*minibatch_share=*/1.0
    );

    Integrator integrator(
        std::make_shared<GaussIntegrand>(integrand),
        trainer,
        std::make_shared<UniformSampler>(sampler),
        /*n_survey_steps=*/10,
        /*n_refine_steps=*/10,
        /*n_points_survey=*/20000,
        /*n_points_refine=*/20000
    );

    integrator.integrate();
}
