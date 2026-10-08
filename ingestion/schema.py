from dataclasses import dataclass, field


@dataclass
class FunctionSample:
    code: str
    cwes: list[str]   # CWE IDs attributed to this function; empty for label=0
    label: int        # 1 = vulnerable, 0 = safe
    branch: str = ""     # "real" | "synth" | "ai"
    language: str = ""   # "C/C++", "Java", "Python"
    sample_id: str = ""  # stable content-derived ID; set at extraction time
