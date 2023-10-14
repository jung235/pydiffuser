import pickle
from typing import Any

from pydiffuser.typing import PathType


def save(obj: Any, pickle_path: PathType) -> None:
    with open(pickle_path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return


def load(pickle_path: PathType) -> Any:
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    return obj


# TODO `jax.vmap` to calculate <O> for multiple lagtimes
