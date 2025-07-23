from pathlib import Path


def get_last_checkpoint_dir(path):
    """
    List all direct child directories of *path* and return the one that is
    alphabetically last. Returns None if the directory has no children.

    Examples
    --------
    >>> get_last_checkpoint_dir('/tmp')  # doctest: +SKIP
    PosixPath('/tmp/z_latest')
    """
    p = Path(path)

    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    child_dirs = [d for d in p.iterdir() if d.is_dir()]
    child_dirs.sort()  # alphabetical, case-sensitive

    return child_dirs[-1] if child_dirs else None
