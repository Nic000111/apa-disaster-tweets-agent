"""Dummy Transformer experiment hook module (slot 3)."""


def preflight_issues(code: str, arch: str) -> list[str]:
    return []


def apply_light_autofixes(code: str, arch: str) -> str:
    return code


def build_repair_hint(stderr_text: str) -> str:
    return ""


def get_arch_prompt() -> str:
    return ""


def get_model_prompt() -> str:
    return ""


def get_template(arch_templates: dict[str, str]) -> str | None:
    return None
