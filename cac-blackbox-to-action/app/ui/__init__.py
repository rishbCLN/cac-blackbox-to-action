"""Demo UI package: page modules and shared view helpers.

Each ``page_*`` module exposes a top-level ``render()`` that Streamlit runs when
its :class:`st.Page` is selected.  Shared state and vocabulary live in
:mod:`app.ui.state` and :mod:`app.ui.theme`; everything data-related is delegated
to the already-tested :mod:`app.components` (stored results) and :mod:`app.live`
(the live engine).
"""

from __future__ import annotations

__all__: list[str] = []
