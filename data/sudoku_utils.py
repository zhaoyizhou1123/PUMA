from omegaconf import OmegaConf, open_dict


def resolve_sudoku_grid_size(cfg):
    """If cfg.grid_size is set, compute and fill in n-dependent sudoku config fields.

    Sets:
        model.vocab_size  = n^2 + 2
        model.max_position = 2 * n^4
        data.mask_id      = n^2 + 1
    """
    if OmegaConf.select(cfg, "grid_size") is None:
        return cfg
    n = cfg.grid_size
    n2 = n ** 2
    n4 = n ** 4
    with open_dict(cfg):
        cfg.model.vocab_size = n2 + 2
        cfg.model.max_position = 2 * n4
        cfg.data.mask_id = n2 + 1
    return cfg
