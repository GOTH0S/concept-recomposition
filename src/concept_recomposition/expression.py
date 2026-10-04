from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["var", "concept", "unary", "binary", "where"]


@dataclass(frozen=True, slots=True)
class Expression:
    kind: Kind
    name: str
    args: tuple["Expression", ...] = ()
    param: int | None = None

    @staticmethod
    def var(name: str) -> "Expression":
        return Expression("var", name)

    @staticmethod
    def concept(name: str) -> "Expression":
        return Expression("concept", name)

    @staticmethod
    def unary(
        name: str,
        arg: "Expression",
        param: int | None = None,
    ) -> "Expression":
        return Expression("unary", name, (arg,), param)

    @staticmethod
    def binary(
        name: str,
        left: "Expression",
        right: "Expression",
    ) -> "Expression":
        if name in {"add", "mul", "min", "max"} and str(left) > str(right):
            left, right = right, left
        return Expression("binary", name, (left, right))

    @staticmethod
    def where(
        condition: "Expression",
        when_positive: "Expression",
        otherwise: "Expression",
    ) -> "Expression":
        return Expression("where", "where", (condition, when_positive, otherwise))

    @property
    def local_depth(self) -> int:
        if not self.args:
            return 0
        return 1 + max(arg.local_depth for arg in self.args)

    @property
    def node_count(self) -> int:
        return 1 + sum(arg.node_count for arg in self.args)

    def operators(self) -> tuple[str, ...]:
        if self.kind in {"var", "concept"}:
            return ()
        found = [self.name]
        for arg in self.args:
            found.extend(arg.operators())
        return tuple(found)

    def __str__(self) -> str:
        if self.kind in {"var", "concept"}:
            return self.name
        if self.kind == "unary":
            suffix = "" if self.param is None else f",{self.param}"
            return f"{self.name}({self.args[0]}{suffix})"
        if self.kind == "binary":
            return f"{self.name}({self.args[0]},{self.args[1]})"
        return f"where({self.args[0]},{self.args[1]},{self.args[2]})"
