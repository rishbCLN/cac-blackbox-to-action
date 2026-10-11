"""Story spine and presenter tooling for the demo (the wayfinding layer).

Where :mod:`app.ui.motion` is pure animation strings, this module owns the
*flow*: the Acts 0--7 step registry, the global stepper rail with Back/Next, the
speed control, the presenter HUD (script line, click list, timing budget and a
stopwatch), the demo reset, and the standardized loading / missing-artifact /
figure-error cards every page routes through.

All session-state keys the interactive layer introduces live here so there is a
single place to clear them (:func:`reset_demo`).  Nothing here trains or queries
a model; it is pure view + navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from app.ui import motion as M

# ------------------------------------------------------------- session keys
K_STEP = "cac_step"            # which Act (0..7) the presenter is on
K_SPEED = "cac_speed"          # motion preset: instant / smooth / cinematic
K_PRESENTER = "cac_presenter"  # presenter HUD visible?
K_REDUCED = "cac_reduced"      # force reduced motion (accessibility)
K_REVEALED = "cac_revealed"    # set of one-shot reveal flags already played


# -------------------------------------------------------------- step registry
@dataclass(frozen=True)
class Step:
    """One Act in the story: its index, short label, nav title, url and page fn."""

    index: int
    key: str
    label: str          # short rail label
    title: str          # sidebar/nav title
    url_path: str
    icon: str
    module: str         # dotted path to the page module
    fn: str = "render"  # render callable name


#: The spine, in order.  ``module``/``fn`` are resolved lazily (see
#: :func:`render_callable`) so this registry imports with no page dependency and
#: cannot create an import cycle.
STEPS: tuple[Step, ...] = (
    Step(0, "cover", "Cover", "Welcome", "cover", "\U0001F3AC",
         "app.ui.page_cover"),
    Step(1, "problem", "Problem", "Home", "home", "\U0001F3E0",
         "app.ui.page_home"),
    Step(2, "gap", "Gap", "The Gap", "gap", "\U0001F4CA",
         "app.ui.page_gap"),
    Step(3, "head", "Head-to-head", "Head to head", "applicant",
         "\u2696\uFE0F", "app.ui.page_applicant"),
    Step(4, "sandbox", "Sandbox", "Live sandbox", "sandbox",
         "\U0001F39B\uFE0F", "app.ui.page_sandbox"),
    Step(5, "map", "Map", "Causal map", "map", "\U0001F578\uFE0F",
         "app.ui.page_map"),
    Step(6, "evidence", "Evidence", "Is this real?", "evidence",
         "\U0001F52C", "app.ui.page_evidence"),
    Step(7, "takeaway", "Takeaway", "Takeaway", "takeaway", "\U0001F3AF",
         "app.ui.page_takeaway"),
)

N_STEPS = len(STEPS)
LAST_STEP = N_STEPS - 1

#: Registered st.Page objects per step index, set by app.py so navigation
#: buttons can call ``st.switch_page``.  Empty when running headless.
_PAGES: dict[int, Any] = {}


def set_pages(pages_by_step: dict[int, Any]) -> None:
    """Register the live ``st.Page`` objects (keyed by step index) from app.py."""
    _PAGES.clear()
    _PAGES.update(pages_by_step)


def render_callable(step: Step) -> Callable[[], None]:
    """Resolve a step's page render function lazily (importlib, no cycle)."""
    import importlib  # noqa: PLC0415

    mod = importlib.import_module(step.module)
    fn = getattr(mod, step.fn)
    if not callable(fn):  # pragma: no cover - registry guards against this
        raise TypeError(f"{step.module}.{step.fn} is not callable")
    return fn


def step_by_key(key: str) -> Step | None:
    return next((s for s in STEPS if s.key == key), None)


# --------------------------------------------------------------- navigation
def current_step() -> int:
    """The Act the presenter is on (session ``K_STEP``), clamped to range."""
    return _clamp(int(st.session_state.get(K_STEP, 0)))


def _clamp(n: int) -> int:
    return max(0, min(int(n), LAST_STEP))


def goto(n: int) -> None:
    """Jump to Act ``n`` (clamped) and navigate there if pages are registered."""
    target = _clamp(n)
    st.session_state[K_STEP] = target
    page = _PAGES.get(target)
    if page is not None:
        try:
            st.switch_page(page)
        except Exception:  # noqa: BLE001 - headless / already on page
            pass


def next_step() -> None:
    goto(current_step() + 1)


def back() -> None:
    goto(current_step() - 1)


# --------------------------------------------------------------- motion state
def speed() -> str:
    """The current motion preset name (session ``K_SPEED``), defaulting to smooth."""
    return M.resolve_speed(st.session_state.get(K_SPEED, M.DEFAULT_SPEED),
                           reduced=reduced())


def reduced() -> bool:
    """Whether reduced motion is forced (accessibility toggle)."""
    return bool(st.session_state.get(K_REDUCED, False))


def inject_motion() -> str:
    """Inject the motion CSS for the current preset; returns the CSS string."""
    return M.inject_motion_css(st.session_state.get(K_SPEED, M.DEFAULT_SPEED),
                               reduced=reduced())


def speed_control(*, container: Any = None, key: str = "_speed_ctl") -> str:
    """Render the speed-preset + reduced-motion controls; return resolved speed.

    Writes ``K_SPEED`` / ``K_REDUCED``.  Uses ``st.segmented_control`` where
    available (snappier than a radio), falling back to a radio otherwise.
    """
    host = container if container is not None else st
    current = st.session_state.get(K_SPEED, M.DEFAULT_SPEED)
    if current not in M.SPEEDS:
        current = M.DEFAULT_SPEED
    labels = [M.SPEED_LABELS[s] for s in M.SPEEDS]
    label_to_key = {M.SPEED_LABELS[s]: s for s in M.SPEEDS}
    picker = getattr(host, "segmented_control", None) or host.radio
    try:
        chosen = picker("Animation speed", labels,
                        default=M.SPEED_LABELS[current], key=key,
                        help="Instant for Q&A, Smooth for a talk, Cinematic for "
                             "the big reveals.")
    except Exception:  # noqa: BLE001 - radio uses index=, older builds vary
        chosen = host.radio("Animation speed", labels,
                            index=list(M.SPEEDS).index(current), key=key)
    chosen_key = label_to_key.get(chosen, M.DEFAULT_SPEED) if chosen else current
    st.session_state[K_SPEED] = chosen_key
    host.toggle("Reduce motion", key=K_REDUCED,
                help="Honour reduced-motion: skip every animation (also forced by "
                     "your OS setting).")
    return M.resolve_speed(chosen_key, reduced=reduced())


# ----------------------------------------------------------- one-shot reveals
def _reveal_set() -> set[str]:
    cur = st.session_state.get(K_REVEALED)
    if not isinstance(cur, set):
        cur = set()
        st.session_state[K_REVEALED] = cur
    return cur


def revealed(flag: str) -> bool:
    """True once ``flag`` has been marked; drives 'play the reveal once' guards."""
    return flag in _reveal_set()


def mark_revealed(flag: str) -> None:
    _reveal_set().add(flag)


def clear_reveal(flag: str) -> None:
    _reveal_set().discard(flag)


# -------------------------------------------------------------- the stepper
_RAIL_CSS_FLAG = "_cac_rail_css"


def _ensure_rail_css() -> None:
    if st.session_state.get(_RAIL_CSS_FLAG):
        return
    st.session_state[_RAIL_CSS_FLAG] = True
    st.markdown(
        f"""<style>
        .cac-rail {{ display:flex; flex-wrap:wrap; gap:0.35rem; align-items:center;
                     margin:0.1rem 0 0.4rem; }}
        .cac-rail .seg {{ font-size:0.78rem; padding:0.12rem 0.5rem; border-radius:999px;
                          border:1px solid {M.COL_BORDER}; color:{M.COL_TEXT_MUTED};
                          background:{M.COL_SURFACE_CARD}; white-space:nowrap; }}
        .cac-rail .seg.done {{ color:{M.COL_PRIMARY}; border-color:{M.COL_PRIMARY}; }}
        .cac-rail .seg.active {{ color:#fff; background:{M.COL_PRIMARY};
                                 border-color:{M.COL_PRIMARY}; font-weight:700; }}
        .cac-rail .arrow {{ color:{M.COL_BORDER}; font-size:0.7rem; }}
        </style>""",
        unsafe_allow_html=True)


def _rail_html(active: int) -> str:
    segs = []
    for s in STEPS:
        cls = "active" if s.index == active else ("done" if s.index < active else "")
        segs.append(f"<span class='seg {cls}'>{s.index}. {s.label}</span>")
        if s.index != LAST_STEP:
            segs.append("<span class='arrow'>&rsaquo;</span>")
    return f"<div class='cac-rail'>{''.join(segs)}</div>"


def stepper(step: int, *, show_rail: bool = True) -> None:
    """Render the global wayfinding at the top of a page and sync ``K_STEP``.

    Shows the 0--7 progress rail plus big Back / Next buttons and "Step n of 7".
    Call once at the top of every page with that page's Act index.
    """
    st.session_state[K_STEP] = _clamp(step)
    _ensure_rail_css()
    if show_rail:
        st.markdown(_rail_html(step), unsafe_allow_html=True)

    back_col, mid_col, next_col = st.columns([1, 2, 1])
    with back_col:
        if step > 0 and st.button("\u25C0 Back", key=f"_nav_back_{step}",
                                  width="stretch"):
            back()
    with mid_col:
        label = STEPS[_clamp(step)].title
        st.markdown(
            f"<div style='text-align:center;color:{M.COL_TEXT_MUTED};"
            f"font-size:0.9rem;padding-top:0.35rem;'>Step {step} of {LAST_STEP}"
            f" &middot; <strong style='color:{M.COL_TEXT};'>{label}</strong></div>",
            unsafe_allow_html=True)
    with next_col:
        if step < LAST_STEP:
            nxt = STEPS[step + 1].label
            if st.button(f"{nxt} \u25B6", key=f"_nav_next_{step}",
                         type="primary", width="stretch"):
                next_step()
        else:
            if st.button("Restart \u21BA", key="_nav_restart",
                         width="stretch"):
                reset_demo()
                goto(0)
    st.divider()


def keyboard_nav() -> None:
    """Best-effort ArrowLeft/ArrowRight -> Back/Next (optional; buttons are the contract)."""
    try:
        import streamlit.components.v1 as components  # noqa: PLC0415

        components.html(
            """<script>
            const doc = window.parent.document;
            if (!doc.__cacKeyNav) {
              doc.__cacKeyNav = true;
              doc.addEventListener('keydown', function(e) {
                if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
                let want = null;
                if (e.key === 'ArrowRight') want = 'next';
                else if (e.key === 'ArrowLeft') want = 'back';
                if (!want) return;
                const btns = doc.querySelectorAll('button');
                for (const b of btns) {
                  const t = (b.innerText || '').trim();
                  if (want === 'next' && t.includes('\u25B6')) { b.click(); break; }
                  if (want === 'back' && t.includes('\u25C0')) { b.click(); break; }
                }
              });
            }
            </script>""",
            height=0)
    except Exception:  # noqa: BLE001 - optional nicety
        pass


# ----------------------------------------------------------- takeaway banner
def banner(sentence: str, *, number: float | None = None, fmt: str = "pct",
           number_label: str = "", color: str = M.COL_PRIMARY) -> None:
    """A per-page 'what the judge should take away' banner with one big number.

    When ``number`` is given it is rendered as an animated count-up at the
    current speed; otherwise just the sentence shows.
    """
    left, right = st.columns([3, 1]) if number is not None else (st, None)
    left.markdown(
        f"<div style='background:{M.COL_PRIMARY}0F;border-left:4px solid {color};"
        f"border-radius:10px;padding:0.6rem 0.95rem;margin:0.2rem 0 0.6rem;"
        f"font-size:1.08rem;font-weight:600;color:{M.COL_TEXT};'>{sentence}</div>",
        unsafe_allow_html=True)
    if number is not None and right is not None:
        try:
            import streamlit.components.v1 as components  # noqa: PLC0415

            with right:
                components.html(
                    M.count_up(number, fmt=fmt, color=color, size="2.0rem",
                               speed=speed(), reduced=reduced()),
                    height=52)
                if number_label:
                    st.caption(number_label)
        except Exception:  # noqa: BLE001
            right.metric(number_label or "", M._fmt_value(number, fmt))


# -------------------------------------------------------- standardized states
def loading_card(message: str = "Loading the model once\u2026",
                 kind: str = "card") -> None:
    """A friendly loading placeholder (skeleton shimmer + caption)."""
    try:
        import streamlit.components.v1 as components  # noqa: PLC0415

        components.html(M.skeleton(kind), height=150)
    except Exception:  # noqa: BLE001
        pass
    st.caption(message)


def missing_card(what: str, *, how: str = "Run `make all` in the project root "
                 "to generate it, then reload.") -> None:
    """A friendly missing-artifact card (never a stack trace)."""
    st.info(f"**{what}** is not available on this machine. {how}")


def safe_pyplot(builder: Callable[..., Any], *args: Any,
                caption: str | None = None, **kwargs: Any) -> bool:
    """Call a matplotlib builder and render it, turning any failure into a caption.

    Returns ``True`` on success.  Guarantees a figure render error shows a
    friendly line instead of a traceback (used to wrap every ``st.pyplot``).
    """
    try:
        fig = builder(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a broken figure must not crash a page
        st.caption(f"(figure unavailable: {exc})")
        return False
    try:
        st.pyplot(fig, width="stretch")
        if caption:
            st.caption(caption)
    except Exception:  # noqa: BLE001 - headless no-op
        pass
    return True


def safe_html(html: str, *, height: int = 120) -> bool:
    """Render a components.html fragment, swallowing failures into a caption."""
    try:
        import streamlit.components.v1 as components  # noqa: PLC0415

        components.html(html, height=height)
        return True
    except Exception as exc:  # noqa: BLE001
        st.caption(f"(animation unavailable: {exc})")
        return False


# ------------------------------------------------------------- presenter HUD
def presenter_enabled() -> bool:
    return bool(st.session_state.get(K_PRESENTER, False))


def _stopwatch_html(budget_s: int) -> str:
    """A self-contained start/stop/reset stopwatch with a budget hint (iframe JS)."""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      html,body {{ margin:0; padding:0; background:transparent;
                   font-family:{M.FONT_SANS}; }}
      .sw {{ display:flex; align-items:center; gap:0.5rem; }}
      .t {{ font-family:{M.FONT_MONO}; font-size:1.3rem; font-weight:700;
            color:{M.COL_TEXT}; min-width:64px; }}
      .t.over {{ color:{M.COL_WARN}; }}
      .bg {{ font-size:0.8rem; color:{M.COL_TEXT_MUTED}; }}
      button {{ font-family:{M.FONT_SANS}; border:1px solid {M.COL_BORDER};
                background:#fff; border-radius:8px; padding:0.15rem 0.6rem;
                cursor:pointer; font-size:0.85rem; }}
    </style></head><body>
    <div class="sw">
      <span class="t" id="t">0:00</span>
      <button id="s">Start</button>
      <button id="r">Reset</button>
      <span class="bg">budget {budget_s}s</span>
    </div>
    <script>
      var el=document.getElementById('t'), sb=document.getElementById('s'),
          rb=document.getElementById('r'), budget={budget_s};
      var running=false, t0=0, acc=0, raf=null;
      function fmt(ms) {{ var s=Math.floor(ms/1000); var m=Math.floor(s/60);
        var r=s%60; return m+':'+(r<10?'0':'')+r; }}
      function tick() {{ var ms=acc+(running?(performance.now()-t0):0);
        el.textContent=fmt(ms); el.className='t'+(ms/1000>budget?' over':'');
        if(running) raf=requestAnimationFrame(tick); }}
      sb.onclick=function(){{ if(running){{acc+=performance.now()-t0;running=false;
        sb.textContent='Start'; if(raf)cancelAnimationFrame(raf); tick();}}
        else {{t0=performance.now();running=true;sb.textContent='Stop';tick();}} }};
      rb.onclick=function(){{running=false;acc=0;sb.textContent='Start';
        if(raf)cancelAnimationFrame(raf);el.textContent='0:00';el.className='t';}};
      tick();
    </script></body></html>"""


def presenter_hud(step: int, *, script: str = "", clicks: list[str] | None = None,
                  watch: str = "", watch_value: Any = None,
                  budget_s: int = 45, next_cue: str = "") -> None:
    """A toggleable rehearsal panel: script, clicks, the number to point at, timer.

    Only renders when the presenter toggle is on (``K_PRESENTER``).  ``watch_value``
    is read live by the caller and shown as the exact on-screen number to point at.
    """
    if not presenter_enabled():
        return
    with st.expander(f"\U0001F3A4 Presenter \u2014 Act {step}: "
                     f"{STEPS[_clamp(step)].title}", expanded=True):
        if script:
            st.markdown(f"**Say:** {script}")
        if clicks:
            st.markdown("**Click:** " + " \u2192 ".join(clicks))
        if watch:
            val = "" if watch_value is None else f"  **{watch_value}**"
            st.markdown(f"**Point at:** {watch}{val}")
        safe_html(_stopwatch_html(int(budget_s)), height=70)
        if next_cue:
            st.caption(f"Next: {next_cue}")


# ---------------------------------------------------------------- reset demo
def reset_demo() -> None:
    """Clear every per-run key so the next walkthrough starts clean.

    Drops the chosen record, all sandbox edits + stale slider widget keys, any
    queued seed features, the step pointer and the one-shot reveal flags.  Keeps
    the dataset, demo-mode, speed and presenter preferences.
    """
    from app.ui import state as S  # noqa: PLC0415 - avoid import cycle at load

    for key in (S.K_RECORD, S.K_SEED, K_STEP, K_REVEALED):
        st.session_state.pop(key, None)
    for k in list(st.session_state.keys()):
        if any(str(k).startswith(p) for p in (
                "_edit_", "_lev_", "_num_", "_imm_", "_seed_", "_recslider_",
                "_sbx_rec_", "_case_", "_appmode_", "_nav_", "_reveal_")):
            st.session_state.pop(k, None)
    st.session_state[K_STEP] = 0


__all__ = [
    "K_STEP", "K_SPEED", "K_PRESENTER", "K_REDUCED", "K_REVEALED",
    "Step", "STEPS", "N_STEPS", "LAST_STEP",
    "set_pages", "render_callable", "step_by_key",
    "current_step", "goto", "next_step", "back",
    "speed", "reduced", "inject_motion", "speed_control",
    "revealed", "mark_revealed", "clear_reveal",
    "stepper", "keyboard_nav", "banner",
    "loading_card", "missing_card", "safe_pyplot", "safe_html",
    "presenter_enabled", "presenter_hud", "reset_demo",
]
