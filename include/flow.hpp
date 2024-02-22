#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

#include "constant.hpp"
#include "coupling.hpp"


class Flow : public torch::nn::Module {
public:
    Flow(
        int64_t dim,
        std::vector<at::Tensor> masks,
        CellType cell_type
    );

    at::Tensor forward(at::Tensor xj);

    void invert();

    // TODO: potential mistake 
    bool is_inverted() { return !cells[0].is_inverted(); }

private:
    int64_t dim;
    std::vector<at::Tensor> masks;
    CellType cell_type;
    std::vector<CouplingCell> cells;
};
