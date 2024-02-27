#include <iostream>
#include <memory>
#include <torch/torch.h>

#include "integrand.hpp"
#include "sampler.hpp"
#include "mask.hpp"
#include "transform.hpp"
#include "trainable.hpp"
#include "flow.hpp"
#include "constant.hpp"
#include "trainer.hpp"
#include "loss.hpp"
#include "integrator.hpp"


void test_transform() {
    PWLinearCouplingTransform transform();

    // at::Tensor x({});

    // transform
}

void test_integration() {
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
        20
    );

    Integrator integrator(
        std::make_shared<GaussIntegrand>(integrand),
        trainer,
        sampler,
        /*n_iter_survey*/10,
        /*n_iter_refine*/10,
        /*n_points_survey*/500,
        /*n_points_refine*/500
    );

    integrator.integrate();
}


int main() {
    test_integration();

    return 0;
}


void test() {
    const int dim = 3;

    GaussIntegrand integrand(dim);
    UniformSampler sampler(dim);
    CheckerboardMask mask(dim);
    PWLinearCouplingTransform transform();
    DNNTrainable trainable(dim, at::tensor({3, 3}), 2, 3);

    std::cout << "Target: " << integrand.target() << std::endl << std::endl;
    std::cout << "Sample: " << sampler.forward(10) << std::endl << std::endl;

    auto masks = mask();

    for (auto m : masks)
        std::cout << "Mask: " << m << std::endl << std::endl;

    PWLinearCouplingCell(
        dim,
        masks[0],
        /*n_bins*/16,
        /*n_hidden*/3,
        /*dim_hidden*/32
    );
    
    Flow flow(dim, masks, CellType::PWLINEAR);

    std::cout << flow.is_inverted() << std::endl;
    flow.invert();
    std::cout << flow.is_inverted() << std::endl;

    Trainer trainer(
        flow,
        sampler,
        variance_loss,
        10
    );

    Integrator integrator(
        std::make_shared<GaussIntegrand>(integrand),
        trainer,
        sampler,
        /*n_iter_survey*/10,
        /*n_iter_refine*/10,
        /*n_points_survey*/2000,
        /*n_points_refine*/2000
    );
}
