"""Thin ArcticDB connection wrapper, shared by the three ArcticDB providers.

``arcticdb`` is a heavy, optional dependency - it is imported lazily, on the
first actual read, so nothing in this package requires it to be installed
unless ``EQR_ARCTICDB_URI`` is actually configured. The library list is
cached after the first connection so repeated reads do not re-list libraries.
"""

from __future__ import annotations

from typing import Any

from ...errors import ProviderError


class ArcticDBConnection:
    """Caches the Arctic connection and its library list for one URI."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._arctic: Any = None
        self._libraries: set[str] | None = None

    def _connect(self) -> Any:
        if self._arctic is None:
            import arcticdb as adb  # local import: optional dependency

            self._arctic = adb.Arctic(self.uri)
            self._libraries = set(self._arctic.list_libraries())
        return self._arctic

    def read(self, library: str, symbol: str) -> Any:
        """Return the DataFrame for ``(library, symbol)``, or ``None`` if absent.

        A missing library or symbol is not an error - most tickers simply are
        not covered by every library - only a genuine connection/read failure
        is raised, and the caller decides how to report that.
        """
        arctic = self._connect()
        if self._libraries is not None and library not in self._libraries:
            return None
        try:
            lib = arctic[library]
            if not lib.has_symbol(symbol):
                return None
            return lib.read(symbol).data
        except Exception as exc:
            raise ProviderError("arcticdb", f"read {library}/{symbol} failed: {exc}") from exc
