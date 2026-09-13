"""Workload preparation contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from torch import Tensor

from tobench.workloads import BaseWorkload

Prepared = TypeVar("Prepared")


class BaseAdapter(ABC, Generic[Prepared]):
    @abstractmethod
    def prepare(self, workload: BaseWorkload, inputs: Sequence[Tensor]) -> Prepared:
        """Return backend-specific state, without compiling an executable."""
        ...
