#include <iostream>
#include <torch/torch.h>

#include "integrand.hpp"
#include "sampler.hpp"
#include "mask.hpp"
#include "transform.hpp"
#include "trainable.hpp"


int main() {
    GaussIntegrand integrand(3);
    UniformSampler sampler(3);
    CheckerboardMask mask(5);
    PWLinearCouplingTransform transform();
    DNNTrainable trainable(3, torch::tensor({3, 3}), 2, 3);

    std::cout << "Target: " << integrand.target() << std::endl << std::endl;

    std::cout << "Sample: " << sampler.forward(10) << std::endl << std::endl;

    for (auto m : mask())
        std::cout << "Mask: " << m << std::endl << std::endl;

    return 0;
}
