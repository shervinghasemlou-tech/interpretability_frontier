
from frontier_interp.config_schema import load_config


def test_debug_config_loads():
    cfg = load_config("configs/debug_qwen.yaml")
    assert cfg.description
    assert len(cfg.models) > 0
    assert len(cfg.datasets) > 0
