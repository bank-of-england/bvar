from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from bvar.models import PosteriorState, SamplingModel, SamplingResult

ADDING_MODELS = Path(__file__).parents[1] / "docs" / "guide" / "adding_models.md"


def _flat_prior_scaffold():
    content = ADDING_MODELS.read_text(encoding="utf-8")
    start = content.index("class FlatPrior(SamplingModel):")
    end = content.index("\n### Wire it up", start)
    scaffold = content[start:end]
    # drop the closing ``` code fence (and anything after it)
    return scaffold[: scaffold.index("\n```")]


def test_flat_prior_scaffold_executes_and_guards():
    scaffold = _flat_prior_scaffold()
    ns: dict = {}
    exec(
        compile(scaffold, "<adding_models.md:FlatPrior>", "exec"),
        {
            "SamplingModel": SamplingModel,
            "SamplingResult": SamplingResult,
            "PosteriorState": PosteriorState,
            "Optional": Optional,
            "np": np,
        },
        ns,
    )
    FlatPrior = ns["FlatPrior"]

    FlatPrior(soc=False, sur=False)
    with pytest.raises(Exception):
        FlatPrior(soc=True)
