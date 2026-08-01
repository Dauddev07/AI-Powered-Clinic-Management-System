"""Validates an email's top-level domain against IANA's official list of registered
TLDs (data.iana.org/TLD/tlds-alpha-by-domain.txt), loaded once at import time — no
network calls at request time. This rejects made-up endings like ".llb" while still
accepting any real TLD (.com, .org, .pk, .io, .co, .edu, and the ~1,500 others),
without depending on whether a specific domain currently has a working mail server.
"""

from pathlib import Path

_TLD_FILE = Path(__file__).parent / "data" / "iana-tlds-alpha-by-domain.txt"


def _load_valid_tlds() -> frozenset[str]:
    tlds = set()
    with _TLD_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tlds.add(line.upper())
    return frozenset(tlds)


VALID_TLDS = _load_valid_tlds()


def is_valid_tld(tld: str) -> bool:
    return tld.upper() in VALID_TLDS
