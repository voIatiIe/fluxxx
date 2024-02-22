#include <iostream>
#include <torch/torch.h>

#include "integrand.hpp"
#include "sampler.hpp"
#include "mask.hpp"
#include "transform.hpp"
#include "trainable.hpp"
#include "flow.hpp"
#include "constant.hpp"


int main() {
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

    return 0;
}
