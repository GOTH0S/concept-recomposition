from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Self

COMMUTATIVE = {"add", "mul", "min", "max"}


class ValueType(str, Enum):
    SERIES = "series"
    BOOLEAN = "boolean"


@dataclass(frozen=True, slots=True)
class Expression:
    op: str
    args: tuple[Expression, ...] = ()
    value: str | None = None
    param: int | None = None
    value_type: ValueType = ValueType.SERIES

    def __post_init__(self) -> None:
        if self.op in {"raw", "concept"}:
            if self.args or self.value is None:
                raise ValueError(f"{self.op} requires a name")
            if self.value_type is not ValueType.SERIES:
                raise TypeError(f"{self.op} is a series")
            return

        if self.op in {"abs", "sign", "rank"}:
            self._require_series_args(1)
            if self.param is not None:
                raise ValueError(f"{self.op} does not take a parameter")
            return

        if self.op in {"lag", "diff", "mean", "std"}:
            self._require_series_args(1)
            if self.param is None or self.param < 1:
                raise ValueError(f"{self.op} requires a positive parameter")
            return

        if self.op in {"add", "sub", "mul", "ratio", "min", "max"}:
            self._require_series_args(2)
            if self.param is not None:
                raise ValueError(f"{self.op} does not take a parameter")
            return

        if self.op == "gt":
            if len(self.args) != 2 or any(
                arg.value_type is not ValueType.SERIES for arg in self.args
            ):
                raise TypeError("gt requires two series")
            if self.value_type is not ValueType.BOOLEAN:
                raise TypeError("gt returns boolean")
            return

        if self.op == "where":
            if len(self.args) != 3:
                raise TypeError("where requires condition, true branch and false branch")
            condition, left, right = self.args
            if condition.value_type is not ValueType.BOOLEAN:
                raise TypeError("where condition must be boolean")
            if left.value_type is not ValueType.SERIES or right.value_type is not ValueType.SERIES:
                raise TypeError("where branches must be series")
            if self.value_type is not ValueType.SERIES:
                raise TypeError("where returns a series")
            return

        raise ValueError(f"unknown operator: {self.op}")

    def _require_series_args(self, count: int) -> None:
        if len(self.args) != count or any(
            arg.value_type is not ValueType.SERIES for arg in self.args
        ):
            raise TypeError(f"{self.op} requires {count} series argument(s)")
        if self.value_type is not ValueType.SERIES:
            raise TypeError(f"{self.op} returns a series")

    @classmethod
    def raw(cls, name: str) -> Self:
        return cls("raw", value=name)

    @classmethod
    def concept(cls, name: str) -> Self:
        return cls("concept", value=name)

    @classmethod
    def unary(cls, op: str, arg: Self, param: int | None = None) -> Self:
        return cls(op, (arg,), param=param)

    @classmethod
    def binary(cls, op: str, left: Self, right: Self) -> Self:
        return cls(op, (left, right))

    @classmethod
    def compare(cls, left: Self, right: Self) -> Self:
        return cls("gt", (left, right), value_type=ValueType.BOOLEAN)

    @classmethod
    def where(cls, condition: Self, left: Self, right: Self) -> Self:
        return cls("where", (condition, left, right))

    @property
    def key(self) -> str:
        if self.op in {"raw", "concept"}:
            return f"{self.op}:{self.value}"
        parts = [arg.key for arg in self.args]
        if self.op in COMMUTATIVE:
            parts.sort()
        suffix = "" if self.param is None else f"[{self.param}]"
        return f"{self.op}{suffix}({','.join(parts)})"

    @property
    def size(self) -> int:
        return 1 + sum(arg.size for arg in self.args)

    @property
    def depth(self) -> int:
        return 0 if not self.args else 1 + max(arg.depth for arg in self.args)

    def concept_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        if self.op == "concept" and self.value is not None:
            ids.append(self.value)
        for arg in self.args:
            ids.extend(arg.concept_ids())
        return tuple(ids)

    def expanded(self, concepts: Mapping[str, Expression]) -> Self:
        if self.op == "concept":
            if self.value not in concepts:
                raise KeyError(f"unknown concept: {self.value}")
            return concepts[self.value].expanded(concepts)
        return Expression(
            self.op,
            tuple(arg.expanded(concepts) for arg in self.args),
            value=self.value,
            param=self.param,
            value_type=self.value_type,
        )

    def __str__(self) -> str:
        if self.op in {"raw", "concept"}:
            return str(self.value)
        suffix = "" if self.param is None else f"[{self.param}]"
        return f"{self.op}{suffix}({', '.join(map(str, self.args))})"
