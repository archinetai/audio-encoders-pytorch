from inspect import isfunction
from typing import Callable, Dict, List, Optional, Sequence, TypeVar, Union

from typing_extensions import TypeGuard

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


"""
Kwargs Utils
"""


def prefix_dict(prefix: str, d: Dict) -> Dict:
    return {prefix + str(k): v for k, v in d.items()}
