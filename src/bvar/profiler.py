import cProfile
import pstats
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def profile_code(
    sort_by: str = "cumtime", top_n: int = 20
) -> Iterator[cProfile.Profile]:
    """
    Context manager for profiling code blocks.

    Parameters
    ----------
    sort_by : str
        Sorting criteria ('cumtime', 'tottime', 'calls', etc.)
    top_n : int
        Number of top functions to display

    Yields
    ------
    cProfile.Profile
        Active profiler for the code block.
    """
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield pr
    finally:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats(sort_by)
        ps.print_stats(top_n)
