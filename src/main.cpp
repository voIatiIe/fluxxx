#include <iostream>
#include <torch/torch.h>

#include "integrand.hpp"
#include "sampler.hpp"


int main() {
    GaussIntegrand integrand(3);
    UniformSampler sampler(3);

    std::cout << "Target: " << integrand.target() << "\n" << std::endl;

    std::cout << "Sample: " << sampler.forward(10) << "\n" << std::endl;
    
    return 0;
}
