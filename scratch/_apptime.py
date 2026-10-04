import os, sys, time, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
from streamlit.testing.v1 import AppTest

ROOT = os.getcwd()
H = (
    'import sys; sys.path.insert(0, r"%s")\n'
    'import streamlit as st\n'
    'from app.ui import state as S\n'
    'st.session_state[S.K_DATASET]="heloc"; st.session_state[S.K_RECORD]=84\n'
    'from %s import render; render()\n'
)
for mod in ["app.ui.page_home", "app.ui.page_sandbox", "app.ui.page_evidence"]:
    t = time.time()
    at = AppTest.from_string(H % (ROOT, mod), default_timeout=120).run()
    print(mod, "EXC" if at.exception else "ok", "err=%d" % len(at.error),
          "%.1fs" % (time.time() - t))
