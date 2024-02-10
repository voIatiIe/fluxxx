#include <iostream>
#include <torch/torch.h>

#include "integrand.hpp"
#include "sampler.hpp"
#include "mask.hpp"


int main() {
    GaussIntegrand integrand(3);
    UniformSampler sampler(3);
    CheckerboardMask mask(5);

    std::cout << "Target: " << integrand.target() << std::endl << std::endl;

    std::cout << "Sample: " << sampler.forward(10) << std::endl << std::endl;
    
    auto masks = mask();

    for (auto m : masks)
        std::cout << "Mask: " << m << std::endl << std::endl;

    return 0;
}
