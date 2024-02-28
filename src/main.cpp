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

void integrate();

int main() {
    integrate();

    return 0;
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
