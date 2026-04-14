
from frontier_interp.config_schema import load_config


def test_debug_config_loads():
    cfg = load_config("configs/debug_qwen.yaml")
    assert cfg.description
    assert len(cfg.models) > 0
    assert len(cfg.datasets) > 0


def test_self_reflection_config_loads():
    cfg = load_config("configs/self_reflection_debug.yaml")
    assert cfg.self_reflection.enabled is True
    assert len(cfg.self_reflection.modes) > 0



def test_full_template_loads_and_supports_reports():
    cfg = load_config("configs/full_paper_template.yaml")
    assert cfg.outputs.save_markdown_report is True
    assert cfg.outputs.save_latex_tables is True
    assert cfg.outputs.save_submission_bundle is True
    assert any(getattr(ds, "use_full_split", False) for ds in cfg.datasets)


def test_submit_ready_balanced_loads():
    cfg = load_config("configs/submit_ready_balanced.yaml")
    assert cfg.description
    assert len(cfg.models) > 0
    assert len(cfg.datasets) > 0
