from __future__ import annotations

from typing import Generic, Sequence, TypeVar


T = TypeVar("T")


class VectorIterator(Generic[T]):
    def __init__(self, vector: Sequence[T]):
        self.vector = vector
        self.index = 0

    def __iter__(self) -> "VectorIterator[T]": # aby šel použít v for-cyklu - new(iterator)
        return self

    def reset(self) -> None:
        self.index = 0

    def __next__(self) -> T:
        if self.index < len(self.vector):
            val = self.vector[self.index]
            self.index += 1
            return val
        else:
            self.reset()
            val = self.vector[self.index]
            self.index += 1
            return val
