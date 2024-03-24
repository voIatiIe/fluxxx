#include <torch/torch.h>

#include <iostream>
#include <cmath>

#include "generator.hpp"
#include "lorentz.hpp"


double PhaseSpaceGenerator::flat_weights(double E, int n) {
    if (n == 1)
        return 1.0;

    return std::pow(2 * M_PI, 4 - 3 * n) * 
           std::pow(M_PI / 2.0, n - 1) * 
          (std::pow(E * E, n - 2) / 
          (std::tgamma(n) * std::tgamma(n - 1)));
}


at::Tensor massless_map(at::Tensor x, at::Tensor exp) {
    return at::pow(x, exp) * ((exp + 1) - exp * x);
}


at::Tensor PhaseSpaceGenerator::bisect_vec_batch(at::Tensor x) {
    const double target = 1e-16;
    int maxLevel = 600;
    int level = 0;

    auto exp = at::arange(n_final-2, 0, -1);

    exp = exp.unsqueeze(0).repeat({x.size(0), 1});

    auto left = at::zeros_like(x);
    auto right = at::ones_like(x);
    auto checkV = -at::ones_like(x);
    auto u = -at::ones_like(x);
    auto error = at::ones_like(x);

    maxLevel /= 10;
    int ml = maxLevel;
    double oldError = 100;

    while (at::max(error).item<double>() > target && ml < 10 * maxLevel) {
        while (level < ml) {
            u = (left + right) * pow(0.5, level + 1);

            checkV = massless_map(u, exp);

            left *= 2.0;
            right *= 2.0;

            auto con = at::full_like(left, 0.5);
            auto adder = at::where(x <= checkV, -con, con);

            left = left + (adder + 0.5);
            right = right + (adder - 0.5);

            level += 1;
        }

        error = at::abs(1.0 - checkV / x);
        ml += maxLevel;

        double newError = at::max(error).item<double>();

        if (newError >= oldError)
            break;
        else
            oldError = newError;
    }

    return u;
}


at::Tensor PhaseSpaceGenerator::generate_massless_batch(at::Tensor M, double E, at::Tensor random_batch) {
    auto u = bisect_vec_batch(random_batch.slice(1, 0, n_final - 2));

    for (int i = 2; i < n_final; ++i)
        M.select(1, i-1) = at::sqrt(u.select(1, i - 2) * at::pow(M.select(1, i - 2), 2));

    return at::tensor({flat_weights(E, n_final)}, torch::kFloat64).expand_as(random_batch.select(1, 0));
}


at::Tensor PhaseSpaceGenerator::rho(at::Tensor M, at::Tensor N, at::Tensor m) {
    auto M2 = at::pow(M, 2);

    return at::sqrt((M2 - at::pow(N + m, 2)) * (M2 - at::pow(N - m, 2))) / (8.0 * M2);
}


at::Tensor PhaseSpaceGenerator::generate_massive_batch(at::Tensor M, double E, at::Tensor random_batch) {
    M.select(1, 0) -= at::sum(masses);

    auto weight = generate_massless_batch(M, E, random_batch);

    auto Mc = M.clone();

    auto masses_sum = at::flip(at::cumsum(at::flip(masses, {0}), 0), {0});
    M += masses_sum.slice(0, 0, masses.size(0) - 1);

    weight = weight * 8.0 * rho(
        M.select(1, n_final - 2),
        masses[n_final - 1],
        masses[n_final - 2]
    );

    weight = weight * at::prod(
        (
            rho(
                M.slice(1, 0, n_final - 2),
                M.slice(1, 1),
                masses.slice(0, 0, n_final - 2)
            )
            /
            rho(
                Mc.slice(1, 0, n_final - 2),
                Mc.slice(1, 1),
                at::zeros_like(Mc.slice(1, 1))
            )
            *
            (M.slice(1, 1, n_final - 1) / Mc.slice(1, 1, n_final - 1))
        ),
        /*dim=*/-1
    );

    weight = weight * at::pow(Mc.select(1, 0) / M.select(1, 0), 2 * n_final - 4);

    return weight;
}


void PhaseSpaceGenerator::set_initial_momenta(at::Tensor& momenta, double E) {
    if (initial_masses[0] == 0.0 || initial_masses[1] == 0.0) {
        auto E_p = E * at::ones({momenta.size(0), 1}, torch::kFloat64) / 2.0;
        auto zeros = at::zeros_like(E_p);

        momenta.select(1, 0) = at::cat({E_p, zeros, zeros, E_p}, 1);
        momenta.select(1, 1) = at::cat({E_p, zeros, zeros, -E_p}, 1);

    } else {
        double M12 = std::pow(initial_masses[0], 2);
        double M22 = std::pow(initial_masses[1], 2);

        auto E1 = (std::pow(E, 2) + M12 - M22) / (2.0 * E);
        auto E2 = (std::pow(E, 2) - M12 + M22) / (2.0 * E);

        double Z = std::sqrt(
            std::pow(E, 4)
            - 2 * std::pow(E, 2) * M12
            - 2 * std::pow(E, 2) * M22
            + std::pow(M12, 2) - 2 * M12 * M22
            + std::pow(M22, 2)
        ) / (2.0 * E);

        momenta.select(1, 0) = at::tensor({E1, 0.0, 0.0, Z}, torch::kFloat64);
        momenta.select(1, 1) = at::tensor({E2, 0.0, 0.0, -Z}, torch::kFloat64);
    }
}


std::tuple<at::Tensor, at::Tensor> PhaseSpaceGenerator::generate_kinematics_batch(
    double E,
    at::Tensor random_batch
) {
    at::Tensor weight_jac = at::ones({random_batch.size(0)}, torch::kFloat64);

    at::Tensor weight = at::ones_like(weight_jac, torch::kFloat64);
    weight *= weight_jac;

    std::vector<double> M_(n_final - 1, 0.0);
    M_[0] = E;

    at::Tensor M = at::tensor(M_, torch::kFloat64);

    M = M.unsqueeze(0).repeat({random_batch.size(0), 1});

    weight *= generate_massive_batch(M, E, random_batch);

    auto Q = at::zeros({1, 4}, torch::kFloat64);
    Q = Q.repeat({random_batch.size(0), 1});
    Q.select(1, 0) = M.select(1, 0);

    M = at::cat({M, masses.unsqueeze(0).repeat({random_batch.size(0), 1}).select(1, -1).unsqueeze(-1)}, -1);

    auto q = 4.0 * M.slice(1, 0, -1)
    * rho(
        M.slice(1, 0, -1),
        M.slice(1, 1),
        masses.slice(0, 0, -1)
    );

    auto rnd = random_batch.slice(1, n_final - 2, 3 * n_final - 4);

    auto cos_theta = 2.0 * rnd.slice(1, 0, rnd.size(1), 2) - 1.0;
    auto sin_theta = at::sqrt(1.0 - at::pow(cos_theta, 2));

    auto phi = 2 * M_PI * rnd.slice(1, 1, rnd.size(1), 2);

    auto cos_phi_t = at::cos(phi);
    auto sin_phi_t = at::where(
        phi > M_PI,
        -at::sqrt(1.0 - at::pow(cos_phi_t, 2)),
        at::sqrt(1.0 - at::pow(cos_phi_t, 2))
    );

    auto a = (q * sin_theta * cos_phi_t).unsqueeze(0);
    auto b = (q * sin_theta * sin_phi_t).unsqueeze(0);
    auto c = (q * cos_theta).unsqueeze(0);


    auto lv = at::cat({at::zeros_like(a), a, b, c}, 0);
    auto result = at::zeros({random_batch.size(0), n_initial + n_final, 4}, torch::kFloat64);


    for (int i = n_initial; i < n_initial + n_final - 1; ++i) {
        auto p2 = lv.select(2, i - n_initial).t();

        p2 = set_square(p2, at::pow(masses[i - n_initial], 2));
        p2 = boost(p2, beta(Q));
        p2 = set_square(p2, at::pow(masses[i - n_initial], 2));

        result.select(1, i) = p2;

        auto Q_ = Q - p2;
        Q_ = set_square(Q_, at::pow(M.select(1, i - n_initial + 1), 2));
        
        Q = Q_;
    }

    result.select(1, -1) = Q;

    set_initial_momenta(result, E);


    auto result_ = result.clone();




    auto pT_mincut = -1.0;
    auto delR_mincut = -1.0;
    auto rap_maxcut = -1.0;




    auto q_theta = std::get<0>(at::min(
        at::sqrt(
            at::pow(result.slice(1, 2).select(2, 1), 2) 
            + at::pow(result.slice(1, 2).select(2, 2), 2)
        ), 
        1
    ));

    auto factor = at::where(
        q_theta < pT_mincut * at::ones_like(q_theta),
        at::zeros_like(weight),
        at::ones_like(weight)
    );

    int final_states = result.size(1) - 2;

    for (int i = 0; i < final_states; ++i) {
        for (int j = 0; j < final_states; ++j) {
            if (i > j) {
                auto delta_r = deltaR(result.select(1, i + 2), result.select(1, j + 2));

                factor *= at::where(
                    at::abs(delta_r) < delR_mincut * at::ones_like(weight),
                    at::zeros_like(weight),
                    at::ones_like(weight)
                );
            }
        }
    }

    if (rap_maxcut > 0) {
        auto rap_max = std::get<0>(at::max(pseudo_rapidity(result.slice(1, 2)), 1));

        factor *= at::where(
            rap_maxcut < at::abs(rap_max),
            at::zeros_like(weight),
            at::ones_like(weight)
        );
    }

    weight *= factor;
    weight /= 2 * E * E;

    return std::make_tuple(result_, weight);
}
