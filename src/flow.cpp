#include <cassert>
#include <torch/torch.h>

#include "flow.hpp"


Flow::Flow(int64_t dim, std::vector<at::Tensor> masks, CellType cell_type) : dim(dim), masks(masks), cell_type(cell_type) {
    assert(masks.size());

    for (int i = 0; i < masks.size(); i++) {
        auto mask = masks[i];

        switch (cell_type) {
        case CellType::PWLINEAR:
            cells -> push_back(
                PWLinearCouplingCell(
                    dim,
                    mask,
                    /*n_bins*/16,
                    /*n_hidden*/3,
                    /*dim_hidden*/32
                )
            );
            break;
        case CellType::PWQUADRATIC:
            // Not implemented yet
            assert(false);
            break;
        default:
            // Not implemented
            assert(false);
            break;
        }
    }

    register_module("cells", cells);
}


// TODO: Make more generic - support PWQuadraticCouplingCell
void Flow::invert() {
    for (auto& cell : *cells) {
        auto cell_ = std::dynamic_pointer_cast<PWLinearCouplingCell>(cell);
        cell_ -> invert();
    }
}


at::Tensor Flow::forward(at::Tensor xj) {
    for (auto& cell : *cells) {
        auto cell_ = std::dynamic_pointer_cast<PWLinearCouplingCell>(cell);
        xj = cell_ -> forward(xj);
    }

    return xj;
}
