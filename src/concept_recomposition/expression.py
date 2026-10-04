from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

COMMUTATIVE = {"add", "mul", "min", "max"}


@dataclass(frozen=True)
class Expression:
    op: str
    args: tuple[Expression, ...] = ()
    value: str | None = None
    param: int | None = None

    @classmethod
    def raw(cls, name: str) -> Expression:
        return cls("raw", value=name)

    @classmethod
    def concept(cls, name: str) -> Expression:
        return cls("concept", value=name)

    @classmethod
    def unary(cls, op: str, arg: Expression, param: int | None = None) -> Expression:
        return cls(op, (arg,), param=param)

    @classmethod
    def binary(cls, op: str, left: Expression, right: Expression) -> Expression:
        return cls(op, (left, right))

    @classmethod
    def where(
        cls, condition: Expression, left: Expression, right: Expression
    ) -> Expression:
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
        return 1 if not self.args else 1 + max(arg.depth for arg in self.args)

    def concept_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        if self.op == "concept" and self.value is not None:
            ids.append(self.value)
        for arg in self.args:
            ids.extend(arg.concept_ids())
        return tuple(ids)

    def subexpression_keys(self) -> set[str]:
        keys = {self.key}
        for arg in self.args:
            keys.update(arg.subexpression_keys())
        return keys

    def expanded(self, concepts: Mapping[str, Expression]) -> Expression:
        if self.op == "concept":
            if self.value not in concepts:
                raise KeyError(f"unknown concept: {self.value}")
            return concepts[self.value].expanded(concepts)
        return Expression(
            self.op,
            tuple(arg.expanded(concepts) for arg in self.args),
            value=self.value,
            param=self.param,
        )

    def __str__(self) -> str:
        if self.op in {"raw", "concept"}:
            return str(self.value)
        suffix = "" if self.param is None else f"[{self.param}]"
        return f"{self.op}{suffix}({', '.join(map(str, self.args))})"
