from pyformlang.cfg import CFG


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    if len(cfg.productions) == 0:
        return cfg

    new_cfg = (
        cfg.remove_useless_symbols()
        .eliminate_unit_productions()
        .remove_useless_symbols()
    )

    new_productions = new_cfg._get_productions_with_only_single_terminals()
    new_productions = new_cfg._decompose_productions(new_productions)
    return CFG(start_symbol=cfg.start_symbol, productions=set(new_productions))
