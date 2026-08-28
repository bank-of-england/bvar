from pathlib import Path

ADDING_MODELS = Path(__file__).parents[1] / "docs" / "guide" / "adding_models.md"


def _flat_prior_scaffold():
    content = ADDING_MODELS.read_text(encoding="utf-8")
    start = content.index("class FlatPrior(SamplingModel):")
    end = content.index("\n### Wire it up", start)
    return content[start:end]


def test_flat_prior_scaffold_guards_unsupported_priors_and_populates_point_only():
    scaffold = _flat_prior_scaffold()

    assert "def __init__(" in scaffold
    assert "soc: bool = False" in scaffold
    assert "sur: bool = False" in scaffold
    assert "if soc or sur:" in scaffold
    assert "FlatPrior does not support soc or sur" in scaffold
    assert "if point_only:" in scaffold
    assert "beta_draws[:] = beta_point" in scaffold
    assert "sigma_draws[:] = sigma_point" in scaffold
