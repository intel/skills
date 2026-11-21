"""Shared skill declarations for Harbor suites and task metadata."""

import re


NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def skill_names(declaration: dict) -> list[str]:
    if "skill" in declaration and "skills" in declaration:
        raise ValueError("declare either 'skill' or 'skills', not both")
    names = [declaration["skill"]] if "skill" in declaration else declaration.get("skills")
    if not isinstance(names, list) or not names:
        raise ValueError("'skills' must be a non-empty list (or use legacy 'skill')")
    if any(not isinstance(name, str) or not NAME.fullmatch(name) for name in names):
        raise ValueError("skill names must be non-empty kebab-case strings")
    if len(set(names)) != len(names):
        raise ValueError("'skills' must not contain duplicates")
    return names


def suite_id(suite: dict) -> str:
    name = suite.get("id", suite.get("skill"))
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError("a suite needs a kebab-case 'id' (or legacy 'skill')")
    return name