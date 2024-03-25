#pragma once

#include <torch/torch.h>

#include <vector>
#include <string>


class PhaseSpaceGenerator {
public:
    PhaseSpaceGenerator(
        std::vector<double> initial_masses_,
        std::vector<double> final_masses_
    ) {
        initial_masses = initial_masses_;
        masses = at::tensor(final_masses_, at::dtype(torch::kFloat64).requires_grad(false));
        n_initial = initial_masses_.size();
        n_final = final_masses_.size();

        if (n_initial != 2)
            throw std::invalid_argument("Only 2 initial particles supported");
    }

    int n_dims() {
        return n_final == 1 ? 0 : 3*n_final - 4;
    }

    std::tuple<at::Tensor, at::Tensor> generate_kinematics_batch(double E, at::Tensor random_batch);

    double flat_weights(double E, int n);

    at::Tensor bisect_vec_batch(at::Tensor x);
    at::Tensor rho(at::Tensor M, at::Tensor N, at::Tensor m);

    at::Tensor generate_massless_batch(at::Tensor M, double E, at::Tensor random_batch);
    at::Tensor generate_massive_batch(at::Tensor M_, double E, at::Tensor random_batch);

    void set_initial_momenta(at::Tensor& output_momenta, double E);

private:
    std::vector<double> initial_masses;
    at::Tensor masses;
    int n_initial;
    int n_final;
};
