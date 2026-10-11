"""Motion system for the demo --- a single source of truth for animation.

Everything here is a **pure string / CSS builder** plus one CSS injector; there
is no ``st.session_state`` logic (that lives in :mod:`app.ui.flow`).  Keeping it
pure means every token, keyframe and HTML builder can be exercised headlessly
by ``scratch/smoke_app.py`` and ``tests/test_motion_flow.py`` without a
Streamlit runtime.

The design rule the whole demo rests on: **fast by default, cinematic on cue.**
Live interaction (slider edits, dataset switches) never animates; the staged
*reveals* that earn the "wow" play once, on a presenter's click.  Three speed
presets gate every duration, and ``prefers-reduced-motion`` (or the Instant
preset) collapses everything to a zero-duration final frame.

Colours and fonts are inlined from the warm light palette so the
``components.v1.html`` iframes (which do not inherit the app theme) match the
rest of the page.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------- palette tokens
# Mirrors app.ui.theme's palette; duplicated (not imported) so motion.py has no
# first-party dependency and stays import-safe in a bare interpreter.
COL_TEXT = "#242220"
COL_TEXT_MUTED = "#5E5953"
COL_SURFACE = "#F7F5F0"
COL_SURFACE_CARD = "#FFFFFF"
COL_BORDER = "#E2DDD2"
COL_PRIMARY = "#2B5C6F"
COL_OK = "#2D6A4F"
COL_OK_BG = "#EBF5EE"
COL_OK_BORDER = "#C2E2CE"
COL_WARN = "#9E2A2B"
COL_WARN_BG = "#FDF0EE"
COL_WARN_BORDER = "#F5C7C4"
COL_CAC = "#1b5e8c"

FONT_SANS = ("'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "
             "'Segoe UI', Roboto, sans-serif")
FONT_MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', monospace"

# ----------------------------------------------------------------- speed presets
#: Valid preset names, slowest-last.  ``smooth`` is the default for a normal talk.
SPEEDS: tuple[str, ...] = ("instant", "smooth", "cinematic")
DEFAULT_SPEED = "smooth"

#: Per-preset durations in milliseconds plus the stagger step and a confetti flag.
#: Instant is all-zero so a skeptic's cross-examination never waits on motion.
SPEED_PRESETS: dict[str, dict[str, Any]] = {
    "instant": {"entrance": 0, "bar": 0, "stagger": 0, "pulse": 0,
                "confetti": False},
    "smooth": {"entrance": 180, "bar": 420, "stagger": 120, "pulse": 220,
               "confetti": False},
    "cinematic": {"entrance": 260, "bar": 650, "stagger": 180, "pulse": 300,
                  "confetti": True},
}

#: Human labels for the speed control (flow.speed_control renders these).
SPEED_LABELS: dict[str, str] = {
    "instant": "Instant",
    "smooth": "Smooth",
    "cinematic": "Cinematic",
}

# easing tokens
EASE_STD = "cubic-bezier(.4,0,.2,1)"
EASE_ENTRANCE = "cubic-bezier(0,0,.2,1)"
EASE_STAMP = "cubic-bezier(.2,.8,.2,1.2)"


def resolve_speed(speed: str | None, reduced: bool = False) -> str:
    """The effective preset: ``prefers-reduced-motion`` always forces Instant.

    Unknown names fall back to the default so a stale session value can never
    break rendering.
    """
    if reduced:
        return "instant"
    key = str(speed or DEFAULT_SPEED).strip().lower()
    return key if key in SPEED_PRESETS else DEFAULT_SPEED


def preset(speed: str | None, reduced: bool = False) -> dict[str, Any]:
    """The resolved duration bundle for a (speed, reduced) pair."""
    return SPEED_PRESETS[resolve_speed(speed, reduced)]


def confetti_enabled(speed: str | None, reduced: bool = False) -> bool:
    """Whether confetti should fire (Cinematic only, never under reduced motion)."""
    return bool(preset(speed, reduced)["confetti"])


def root_class(speed: str | None, reduced: bool = False) -> str:
    """The wrapper class name for the resolved preset (``motion-smooth`` etc.)."""
    return f"motion-{resolve_speed(speed, reduced)}"


# --------------------------------------------------------------------- keyframes
#: Every keyframe name the app references; kept as a tuple so tests can assert
#: the injected CSS defines each one.
KEYFRAMES: tuple[str, ...] = (
    "kf-fade-up", "kf-chip-pop", "kf-shake", "kf-stamp", "kf-grow-x",
    "kf-pulse", "kf-shimmer", "kf-ripple",
)

_KEYFRAME_CSS = """
@keyframes kf-fade-up { from { opacity: 0; transform: translateY(10px); }
                        to   { opacity: 1; transform: translateY(0); } }
@keyframes kf-chip-pop { 0% { opacity: 0; transform: scale(.8); }
                         70% { opacity: 1; transform: scale(1.06); }
                         100% { opacity: 1; transform: scale(1); } }
@keyframes kf-shake { 0%,100% { transform: translateX(0); }
                      20% { transform: translateX(-5px); }
                      40% { transform: translateX(5px); }
                      60% { transform: translateX(-3px); }
                      80% { transform: translateX(3px); } }
@keyframes kf-stamp { 0% { opacity: 0; transform: scale(1.4) rotate(-3deg); }
                      60% { opacity: 1; transform: scale(.94) rotate(0deg); }
                      100% { opacity: 1; transform: scale(1); } }
@keyframes kf-grow-x { from { transform: scaleX(0); }
                       to   { transform: scaleX(1); } }
@keyframes kf-pulse { 0% { transform: scale(1); }
                      40% { transform: scale(1.12); }
                      100% { transform: scale(1); } }
@keyframes kf-shimmer { 0% { background-position: -400px 0; }
                        100% { background-position: 400px 0; } }
@keyframes kf-ripple { 0% { box-shadow: 0 0 0 0 rgba(27,94,140,.45); }
                       100% { box-shadow: 0 0 0 18px rgba(27,94,140,0); } }
"""


def motion_css(speed: str | None, reduced: bool = False) -> str:
    """The full motion stylesheet for one preset, as a string.

    Defines every keyframe, the per-preset duration CSS variables, the utility
    classes that bind keyframes to those variables, and the
    ``prefers-reduced-motion`` hard override.  Pure: returns the string and
    touches no Streamlit state, so tests can assert on its contents.
    """
    p = preset(speed, reduced)
    cls = root_class(speed, reduced)
    entrance = f"{int(p['entrance'])}ms"
    bar = f"{int(p['bar'])}ms"
    stagger = f"{int(p['stagger'])}ms"
    pulse = f"{int(p['pulse'])}ms"

    return f"""<style>
{_KEYFRAME_CSS}
.{cls} {{
  --mo-entrance: {entrance};
  --mo-bar: {bar};
  --mo-stagger: {stagger};
  --mo-pulse: {pulse};
  --mo-ease: {EASE_STD};
  --mo-ease-entrance: {EASE_ENTRANCE};
  --mo-ease-stamp: {EASE_STAMP};
}}
.{cls} .mo-fade-up {{ animation: kf-fade-up var(--mo-entrance) var(--mo-ease-entrance) both; }}
.{cls} .mo-chip-pop {{ animation: kf-chip-pop var(--mo-entrance) var(--mo-ease-stamp) both; }}
.{cls} .mo-shake {{ animation: kf-shake 420ms var(--mo-ease) both; }}
.{cls} .mo-stamp {{ animation: kf-stamp var(--mo-entrance) var(--mo-ease-stamp) both; }}
.{cls} .mo-grow-x {{ transform-origin: left center; animation: kf-grow-x var(--mo-bar) var(--mo-ease-entrance) both; }}
.{cls} .mo-pulse {{ animation: kf-pulse var(--mo-pulse) var(--mo-ease) both; }}
.{cls} .mo-ripple {{ animation: kf-ripple 900ms var(--mo-ease) both; }}
/* staggered children: each subsequent .mo-stagger item waits one step more */
.{cls} .mo-stagger:nth-child(1) {{ animation-delay: calc(var(--mo-stagger) * 0); }}
.{cls} .mo-stagger:nth-child(2) {{ animation-delay: calc(var(--mo-stagger) * 1); }}
.{cls} .mo-stagger:nth-child(3) {{ animation-delay: calc(var(--mo-stagger) * 2); }}
.{cls} .mo-stagger:nth-child(4) {{ animation-delay: calc(var(--mo-stagger) * 3); }}
.{cls} .mo-stagger:nth-child(5) {{ animation-delay: calc(var(--mo-stagger) * 4); }}
.{cls} .mo-stagger:nth-child(6) {{ animation-delay: calc(var(--mo-stagger) * 5); }}
@media (prefers-reduced-motion: reduce) {{
  .{cls} *, .{cls} *::before, .{cls} *::after {{
    animation: none !important;
    transition: none !important;
  }}
}}
</style>"""


def inject_motion_css(speed: str | None, reduced: bool = False) -> str:
    """Inject the motion stylesheet into the page and return it.

    Best-effort ``st.markdown`` (a no-op without a script run), so callers get
    the string back for tests either way.
    """
    css = motion_css(speed, reduced)
    try:
        import streamlit as st  # noqa: PLC0415

        st.markdown(css, unsafe_allow_html=True)
    except Exception:  # noqa: BLE001 - headless / no runtime
        pass
    return css


def open_motion_root(speed: str | None, reduced: bool = False) -> str:
    """Opening ``<div>`` tag carrying the resolved preset's root class."""
    return f"<div class='{root_class(speed, reduced)}'>"


# --------------------------------------------------------- HTML/JS iframe builders

def _fmt_value(value: float, fmt: str) -> str:
    """Format a number the way the on-screen final frame should read."""
    try:
        if fmt == "pct":
            return f"{float(value) * 100:.0f}%"
        if fmt == "pct1":
            return f"{float(value) * 100:.1f}%"
        if fmt == "int":
            return f"{int(round(float(value)))}"
        if fmt == "float2":
            return f"{float(value):.2f}"
        return str(value)
    except (TypeError, ValueError):
        return str(value)


def count_up(value: float, fmt: str = "pct", color: str = COL_PRIMARY,
             size: str = "2.6rem", duration: int = 650,
             *, speed: str | None = None, reduced: bool = False,
             prefix: str = "", suffix: str = "") -> str:
    """A headline number that ticks ``0 -> value`` on mount via ``requestAnimationFrame``.

    Returns a self-contained HTML fragment for ``st.components.v1.html``.  The
    final formatted value is always embedded in the markup (so a reduced-motion
    or Instant render --- ``duration`` 0 --- simply shows it with no tween, and
    tests can assert the live number is present).
    """
    if speed is not None:
        duration = int(preset(speed, reduced)["bar"])
    final = _fmt_value(value, fmt)
    dur = max(int(duration), 0)
    import math  # noqa: PLC0415

    raw = float(value) if _is_number(value) else 0.0
    if not math.isfinite(raw):  # NaN/inf would emit invalid JS; tween from 0
        raw = 0.0
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
  html,body {{ margin:0; padding:0; background:transparent; }}
  .cu {{ font-family:{FONT_MONO}; font-weight:700; color:{color};
         font-size:{size}; line-height:1.1; letter-spacing:-0.01em; }}
</style></head><body>
<span class="cu" id="cu" data-final="{final}">{prefix}{final}{suffix}</span>
<script>
  (function() {{
    var el = document.getElementById('cu');
    var target = {raw!r};
    var dur = {dur};
    var fmt = {fmt!r};
    var prefix = {prefix!r}, suffix = {suffix!r};
    function render(v) {{
      var s;
      if (fmt === 'pct') s = Math.round(v*100) + '%';
      else if (fmt === 'pct1') s = (v*100).toFixed(1) + '%';
      else if (fmt === 'int') s = Math.round(v).toString();
      else if (fmt === 'float2') s = v.toFixed(2);
      else s = v.toString();
      el.textContent = prefix + s + suffix;
    }}
    if (dur <= 0) {{ el.textContent = prefix + el.getAttribute('data-final') + suffix; return; }}
    var start = null;
    function step(ts) {{
      if (start === null) start = ts;
      var t = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - t, 3);
      render(target * eased);
      if (t < 1) requestAnimationFrame(step);
      else el.textContent = prefix + el.getAttribute('data-final') + suffix;
    }}
    requestAnimationFrame(step);
  }})();
</script></body></html>"""


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def gap_bars_html(rows: list[dict[str, Any]], speed: str | None = None,
                  reduced: bool = False, *, height: int = 300) -> str:
    """The hero validity-vs-usable bars for one dataset, as animated CSS/HTML.

    ``rows`` is an ordered list of dicts with keys ``label`` (method name),
    ``validity`` and ``usable`` (0..1 floats), and optional ``is_cac`` /
    ``is_worst`` flags.  Each row draws a faint full-width validity track with a
    solid ``usable`` fill that grows ``0 -> pct`` and a number that counts up;
    the worst baseline is flagged and the CAC row is emphasised.  Pure: every
    live percentage is embedded in the returned string.
    """
    p = preset(speed, reduced)
    bar_ms = int(p["bar"])
    stagger_ms = int(p["stagger"])
    animate = bar_ms > 0

    items = []
    for i, row in enumerate(rows):
        label = str(row.get("label", ""))
        usable = float(row.get("usable", 0.0) or 0.0)
        validity = float(row.get("validity", 0.0) or 0.0)
        is_cac = bool(row.get("is_cac"))
        is_worst = bool(row.get("is_worst"))
        fill = COL_CAC if is_cac else (COL_WARN if is_worst else COL_TEXT_MUTED)
        pct = f"{usable * 100:.0f}%"
        vpct = f"{validity * 100:.0f}%"
        delay = (stagger_ms * i) if animate else 0
        grow = (f"transform:scaleX(0);transform-origin:left;"
                f"animation:gb-grow {bar_ms}ms {EASE_ENTRANCE} {delay}ms forwards;"
                if animate else "transform:scaleX(1);")
        flag = (" <span style='color:%s;font-weight:700;'>&larr; worst baseline</span>" % COL_WARN
                if is_worst else "")
        emph = ("font-weight:800;color:%s;" % COL_CAC) if is_cac else ("color:%s;" % COL_TEXT)
        items.append(f"""
      <div class="gb-row">
        <div class="gb-label" style="{emph}">{label}{flag}</div>
        <div class="gb-track" title="flips the box: {vpct}">
          <div class="gb-validity" style="width:{vpct};"></div>
          <div class="gb-usable" style="width:{pct};{grow}background:{fill};"></div>
          <div class="gb-num" style="color:{fill};" data-final="{pct}">{pct}</div>
        </div>
      </div>""")

    counter_js = f"""
    (function() {{
      var dur = {bar_ms};
      var nums = document.querySelectorAll('.gb-num');
      if (dur <= 0) {{ nums.forEach(function(n){{ n.textContent = n.getAttribute('data-final'); }}); return; }}
      nums.forEach(function(n, i) {{
        var finalPct = parseFloat(n.getAttribute('data-final'));
        var delay = {stagger_ms} * i;
        var start = null;
        function step(ts) {{
          if (start === null) start = ts;
          var e = ts - start - delay;
          if (e < 0) {{ requestAnimationFrame(step); return; }}
          var t = Math.min(e / dur, 1);
          var eased = 1 - Math.pow(1 - t, 3);
          n.textContent = Math.round(finalPct * eased) + '%';
          if (t < 1) requestAnimationFrame(step);
          else n.textContent = n.getAttribute('data-final');
        }}
        requestAnimationFrame(step);
      }});
    }})();
    """

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
  html,body {{ margin:0; padding:0; background:transparent; font-family:{FONT_SANS}; }}
  @keyframes gb-grow {{ to {{ transform: scaleX(1); }} }}
  .gb-row {{ margin:0.55rem 0; }}
  .gb-label {{ font-size:0.95rem; margin-bottom:0.2rem; }}
  .gb-track {{ position:relative; height:28px; background:{COL_SURFACE};
               border:1px solid {COL_BORDER}; border-radius:8px; overflow:hidden; }}
  .gb-validity {{ position:absolute; left:0; top:0; height:100%;
                  background:{COL_BORDER}; opacity:0.6; }}
  .gb-usable {{ position:absolute; left:0; top:0; height:100%; border-radius:0 6px 6px 0; }}
  .gb-num {{ position:absolute; right:8px; top:50%; transform:translateY(-50%);
             font-family:{FONT_MONO}; font-weight:700; font-size:0.9rem; }}
  @media (prefers-reduced-motion: reduce) {{ .gb-usable {{ animation:none !important; transform:scaleX(1) !important; }} }}
</style></head><body>
<div class="gb-wrap">{''.join(items)}</div>
<script>{counter_js}</script>
</body></html>"""


def verdict_stamp_html(usable: bool, speed: str | None = None,
                       reduced: bool = False, *, height: int = 90) -> str:
    """A big USABLE / NOT USABLE badge that stamps in (overshoot) with a soft shadow."""
    p = preset(speed, reduced)
    dur = int(p["entrance"])
    animate = dur > 0
    bg = COL_OK if usable else COL_WARN
    text = "USABLE" if usable else "NOT USABLE"
    icon = "\u2713" if usable else "\u2715"
    shadow = ("rgba(45,106,79,0.30)" if usable else "rgba(158,42,43,0.30)")
    anim = (f"animation:vs-stamp {dur}ms {EASE_STAMP} both;" if animate else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
  html,body {{ margin:0; padding:0; background:transparent; font-family:{FONT_SANS}; }}
  @keyframes vs-stamp {{ 0% {{ opacity:0; transform:scale(1.4) rotate(-3deg); }}
                         60% {{ opacity:1; transform:scale(.94); }}
                         100% {{ opacity:1; transform:scale(1); }} }}
  .vs {{ display:inline-flex; align-items:center; gap:0.5rem; background:{bg};
         color:#fff; border-radius:999px; padding:0.5rem 1.4rem; {anim}
         font-size:1.5rem; font-weight:800; letter-spacing:0.05em;
         box-shadow:0 6px 18px {shadow}; }}
  .vs .ic {{ font-size:1.3rem; }}
  @media (prefers-reduced-motion: reduce) {{ .vs {{ animation:none !important; }} }}
</style></head><body>
<span class="vs"><span class="ic">{icon}</span>{text}</span>
</body></html>"""


def confetti_html(*, height: int = 220, color_a: str = COL_OK,
                  color_b: str = COL_CAC, color_c: str = "#e0a81a") -> str:
    """A tiny self-contained canvas confetti burst (no dependencies).

    Only the caller decides when to show this (Cinematic + demo mode on a USABLE
    verdict); it always plays once on mount.
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
  html,body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  canvas {{ display:block; width:100%; height:100%; }}
</style></head><body>
<canvas id="cf"></canvas>
<script>
  (function() {{
    var c = document.getElementById('cf');
    var ctx = c.getContext('2d');
    function size() {{ c.width = c.offsetWidth; c.height = {height}; }}
    size();
    var colors = [{color_a!r}, {color_b!r}, {color_c!r}, '#ffffff'];
    var N = 90, parts = [];
    for (var i = 0; i < N; i++) {{
      parts.push({{
        x: c.width/2 + (Math.random()-0.5)*80,
        y: c.height*0.35,
        vx: (Math.random()-0.5)*9,
        vy: Math.random()*-9 - 3,
        g: 0.28 + Math.random()*0.12,
        s: 4 + Math.random()*5,
        col: colors[(Math.random()*colors.length)|0],
        rot: Math.random()*Math.PI, vr: (Math.random()-0.5)*0.3
      }});
    }}
    var t = 0, max = 110;
    function frame() {{
      ctx.clearRect(0,0,c.width,c.height);
      parts.forEach(function(p) {{
        p.x += p.vx; p.y += p.vy; p.vy += p.g; p.rot += p.vr;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot);
        ctx.fillStyle = p.col; ctx.fillRect(-p.s/2, -p.s/2, p.s, p.s*0.6);
        ctx.restore();
      }});
      t++;
      if (t < max) requestAnimationFrame(frame);
      else ctx.clearRect(0,0,c.width,c.height);
    }}
    requestAnimationFrame(frame);
  }})();
</script></body></html>"""


_SKELETON_KINDS = ("card", "bars", "line")


def skeleton(kind: str = "card", *, height: int = 120) -> str:
    """A shimmer placeholder for loading states (``card`` / ``bars`` / ``line``)."""
    kind = kind if kind in _SKELETON_KINDS else "card"
    if kind == "bars":
        blocks = "".join(
            f"<div class='sk-bar' style='width:{w}%;'></div>"
            for w in (90, 70, 80, 55))
    elif kind == "line":
        blocks = "<div class='sk-bar' style='width:70%;'></div>"
    else:
        blocks = ("<div class='sk-bar' style='width:60%;height:1.3rem;'></div>"
                  "<div class='sk-bar' style='width:95%;'></div>"
                  "<div class='sk-bar' style='width:85%;'></div>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
  html,body {{ margin:0; padding:0; background:transparent; font-family:{FONT_SANS}; }}
  .sk {{ background:{COL_SURFACE_CARD}; border:1px solid {COL_BORDER};
         border-radius:14px; padding:1rem 1.1rem; }}
  .sk-bar {{ height:0.9rem; border-radius:6px; margin:0.5rem 0;
             background:linear-gradient(90deg,#eee 25%,#f6f4f0 37%,#eee 63%);
             background-size:800px 100%; animation:kf-shimmer 1.4s linear infinite; }}
  @keyframes kf-shimmer {{ 0% {{ background-position:-400px 0; }}
                           100% {{ background-position:400px 0; }} }}
  @media (prefers-reduced-motion: reduce) {{ .sk-bar {{ animation:none !important; }} }}
</style></head><body><div class="sk">{blocks}</div></body></html>"""


__all__ = [
    "SPEEDS", "DEFAULT_SPEED", "SPEED_PRESETS", "SPEED_LABELS", "KEYFRAMES",
    "resolve_speed", "preset", "confetti_enabled", "root_class",
    "motion_css", "inject_motion_css", "open_motion_root",
    "count_up", "gap_bars_html", "verdict_stamp_html", "confetti_html",
    "skeleton",
]
