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
#include "wrapper.hpp"

void integrate();

int main() {
    std::vector<double> initial_masses = {0.0, 0.0};
    std::vector<double> final_masses = {0.0, 0.0, 0.0, 0.0};

    PhaseSpaceGenerator generator(initial_masses, final_masses);

    auto result = generator.generate_kinematics_batch(1000.0, torch::tensor({{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}}));

    std::cout << std::get<0>(result) << std::endl;
    std::cout << std::get<1>(result) << std::endl;

    return 0;
}


void python_call() {
    at::Tensor tensor = torch::tensor({
        {1.0, 2.0, 1.0, 2.0},
        {3.0, 4.0, 1.0, 2.0},
        {5.0, 6.0, 1.0, 2.0},
        {7.0, 8.0, 1.0, 2.0}
    }).to(at::kFloat);

    MatrixWrapper wrapper;

    wrapper.initialisemodel();
    double result = wrapper.smatrix(tensor);

    std::cout << result << std::endl;
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
        sampler,
        variance_loss,
        /*n_epochs=*/10,
        /*minibatch_share=*/1.0
    );

    Integrator integrator(
        std::make_shared<GaussIntegrand>(integrand),
        trainer,
        sampler,
        /*n_survey_steps=*/10,
        /*n_refine_steps=*/10,
        /*n_points_survey=*/20000,
        /*n_points_refine=*/20000
    );

    integrator.integrate();
}
