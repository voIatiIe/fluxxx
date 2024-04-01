#include <memory>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "mpi.h"

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


std::pair<double, double> integrate_mg();


int main(int32_t argc, char** argv) {
    MPI_Init(&argc, &argv);

    int32_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto res = integrate_mg();
    auto var = res.second * res.second;

    double total_estimate = 0.0;
    double total_variance = 0.0;

    MPI_Reduce(&res.first, &total_estimate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&var, &total_variance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double average_estimate = total_estimate / size;
        double average_variance = std::sqrt(total_variance) / size;

        std::cout << "Estimate: " << average_estimate << " +- " << average_variance << std::endl;
    }

    MPI_Finalize();

    return 0;
}


std::pair<double, double> integrate_mg() {
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
        dkl_loss,
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

    return integrator.integrate();
}
