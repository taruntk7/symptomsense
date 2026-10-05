import html as htmllib
import re
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

from backend.pipeline import predict_case


# ============================================================
# Constants
# ============================================================

# (minimum percent, css key, label, note shown under the prediction)
CONFIDENCE_LEVELS = [
    (
        70,
        "high",
        "High",
        "The model is fairly confident in this prediction."
    ),
    (
        40,
        "moderate",
        "Moderate",
        "The model leans this way, but other conditions remain plausible."
    ),
    (
        0,
        "low",
        "Low",
        "The model is uncertain. Treat this as one of several possibilities."
    ),
]

# ------------------------------------------------------------
# Colour palettes. The whole UI is driven by these variables, so
# switching theme only swaps this block. "pos", "neg" and "warn"
# are RGB triplets so they can be used with any alpha.
# ------------------------------------------------------------

THEMES = {
    "light": {
        "scheme": "light",
        "bg": "#f6f5f1",
        "surface": "#ffffff",
        "field": "#ffffff",
        "hover": "#efede7",
        "text": "#1c1f23",
        "muted": "#5b626b",
        "faint": "#6d747e",
        "border": "#dedbd3",
        "strong": "#c5c2b8",
        "track": "#e8e6e0",
        "accent": "#1a7f4b",
        "accent_hover": "#146a3e",
        "accent_text": "#1a7f4b",
        "on_accent": "#ffffff",
        "pos": "31, 138, 91",
        "neg": "196, 71, 47",
        "warn": "168, 107, 10",
    },
    "dark": {
        "scheme": "dark",
        "bg": "#111315",
        "surface": "#171a1d",
        "field": "#171a1d",
        "hover": "#1e2226",
        "text": "#e7e5e0",
        "muted": "#a3a8b0",
        "faint": "#7b818a",
        "border": "#2b3035",
        "strong": "#3c424a",
        "track": "#262b30",
        "accent": "#1f8a54",
        "accent_hover": "#25a062",
        "accent_text": "#4cc38a",
        "on_accent": "#ffffff",
        "pos": "76, 195, 138",
        "neg": "239, 127, 104",
        "warn": "224, 165, 60",
    },
}

def _svg_uri(inner: str) -> str:
    """
    Build a URL-encoded data URI for a stroke icon. Inline <svg> is
    stripped by Streamlit's HTML sanitizer, and a raw '<' inside a style
    block makes the sanitizer drop the whole block, so icons are drawn as
    CSS masks with the markup percent-encoded.
    """
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        "fill='none' stroke='black' stroke-width='1.8' "
        "stroke-linecap='round' stroke-linejoin='round'>"
        f"{inner}</svg>"
    )
    return f'url("data:image/svg+xml,{quote(svg)}")'


ICON_PULSE = _svg_uri("<path d='M3 12h4l2.5-6 4 12 2.5-6H21'/>")

ICON_INFO = _svg_uri(
    "<circle cx='12' cy='12' r='9'/>"
    "<path d='M12 11v5'/><path d='M12 8h.01'/>"
)

MARK_SVG = '<span class="ss-ico ss-ico-pulse"></span>'
INFO_SVG = '<span class="ss-ico ss-ico-info"></span>'


# Language used for speech recognition (BCP-47). "en-IN" suits Indian
# English accents; use "en-US", "en-GB", "hi-IN", etc. to change it.
SPEECH_LANG = "en-IN"

# ------------------------------------------------------------
# Voice input. Speech recognition runs in the browser (Web Speech
# API), so it needs Chrome, Edge or Safari. Streamlit cannot run
# JavaScript in st.html, so the button lives in a small iframe that
# writes the transcript straight into the page's text box.
# ------------------------------------------------------------

VOICE_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="__SCHEME__">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    color-scheme: __SCHEME__;
    __VARS__
}

html, body {
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system,
                 "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
}

.row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    height: 40px;
}

button {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    height: 36px;
    padding: 0 0.95rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    color: var(--text);
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.12s ease, border-color 0.12s ease;
}

button:hover:not(:disabled) {
    background: var(--hover);
    border-color: var(--strong);
}

button:focus-visible {
    outline: 2px solid var(--accent-text);
    outline-offset: 2px;
}

button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
}

button svg {
    width: 16px;
    height: 16px;
}

.dot {
    display: none;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgb(var(--neg));
    animation: blink 1.2s ease-in-out infinite;
}

button[aria-pressed="true"] {
    border-color: rgba(var(--neg), 0.6);
}

button[aria-pressed="true"] .dot {
    display: inline-block;
}

button[aria-pressed="true"] svg {
    display: none;
}

.status {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.3;
}

.status.error {
    color: rgb(var(--neg));
}

@keyframes blink {
    50% { opacity: 0.25; }
}

@media (prefers-reduced-motion: reduce) {
    .dot { animation: none; }
}
</style>
</head>
<body>

<div class="row">
    <button id="mic" type="button" aria-pressed="false"
            aria-label="Dictate symptoms with your voice">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.8" stroke-linecap="round"
             stroke-linejoin="round" aria-hidden="true">
            <rect x="9" y="3" width="6" height="11" rx="3"></rect>
            <path d="M5 11a7 7 0 0 0 14 0"></path>
            <path d="M12 18v3"></path>
        </svg>
        <span class="dot"></span>
        <span id="label">Dictate</span>
    </button>
    <span id="status" class="status" role="status" aria-live="polite">
        Or speak your symptoms and edit the text before analyzing.
    </span>
</div>

<script>
(function () {
    var LANG = "__LANG__";

    var btn = document.getElementById("mic");
    var label = document.getElementById("label");
    var statusEl = document.getElementById("status");
    var Rec = window.SpeechRecognition || window.webkitSpeechRecognition;

    function setStatus(message, isError) {
        statusEl.textContent = message;
        statusEl.className = "status" + (isError ? " error" : "");
    }

    if (!Rec) {
        btn.disabled = true;
        setStatus("Voice input is not supported in this browser. " +
                  "Try Chrome, Edge or Safari.", true);
        return;
    }

    var rec = null;
    var listening = false;
    var base = "";
    var box = null;
    var gotSpeech = false;
    var errored = false;

    function findBox() {
        try {
            var doc = window.parent.document;
            return doc.querySelector('[data-testid="stTextArea"] textarea') ||
                   doc.querySelector("textarea");
        } catch (e) {
            return null;
        }
    }

    // Write into a React-controlled textarea so Streamlit sees the change.
    function setValue(el, value) {
        var win = window.parent;
        var setter = Object.getOwnPropertyDescriptor(
            win.HTMLTextAreaElement.prototype, "value").set;
        setter.call(el, value);
        el.dispatchEvent(new win.Event("input", { bubbles: true }));
    }

    function compose(spoken) {
        spoken = spoken.replace(/\s+/g, " ").trim();
        if (!spoken) { return base; }
        if (!base.trim()) {
            return spoken.charAt(0).toUpperCase() + spoken.slice(1);
        }
        return base.replace(/\s+$/, "") + " " + spoken;
    }

    function markListening(on) {
        if (!box) { return; }
        var wrap = box.closest('[data-testid="stTextAreaRootElement"]') ||
                   box.closest('[data-baseweb="textarea"]');
        if (!wrap) { return; }
        if (on) { wrap.setAttribute("data-listening", "true"); }
        else { wrap.removeAttribute("data-listening"); }
    }

    // Streamlit sends the text to Python when the box loses focus.
    function commit() {
        try {
            box.focus({ preventScroll: true });
            box.blur();
        } catch (e) {}
    }

    function explain(code) {
        if (code === "not-allowed" || code === "service-not-allowed") {
            return "Microphone access is blocked. Allow it in the browser's " +
                   "site settings (needs HTTPS or localhost).";
        }
        if (code === "no-speech") { return "Did not hear anything. Try again."; }
        if (code === "audio-capture") { return "No microphone was found."; }
        if (code === "network") {
            return "Speech service unreachable. Voice input needs internet.";
        }
        if (code === "language-not-supported") {
            return "This language is not supported by your browser.";
        }
        return "Voice input error (" + code + ").";
    }

    function onUserTyping() { stop(); }

    function stop() {
        if (rec && listening) { rec.stop(); }
    }

    function start() {
        if (window.isSecureContext === false) {
            setStatus("Voice input needs HTTPS or localhost.", true);
            return;
        }

        box = findBox();
        if (!box) { setStatus("Could not find the text box.", true); return; }

        base = box.value || "";
        gotSpeech = false;
        errored = false;

        rec = new Rec();
        rec.lang = LANG;
        rec.interimResults = true;
        rec.maxAlternatives = 1;
        rec.continuous = !/Android/i.test(navigator.userAgent);

        rec.onstart = function () {
            listening = true;
            btn.setAttribute("aria-pressed", "true");
            label.textContent = "Stop";
            setStatus("Listening... speak now.");
            markListening(true);
            box.addEventListener("keydown", onUserTyping);
        };

        rec.onresult = function (event) {
            var finalText = "";
            var interimText = "";
            for (var i = 0; i < event.results.length; i++) {
                var result = event.results[i];
                if (result.isFinal) { finalText += result[0].transcript; }
                else { interimText += result[0].transcript; }
            }
            gotSpeech = true;
            setValue(box, compose(finalText + " " + interimText));
            box.scrollTop = box.scrollHeight;
        };

        rec.onerror = function (event) {
            if (event.error === "aborted") { return; }
            errored = true;
            setStatus(explain(event.error), true);
        };

        rec.onend = function () {
            listening = false;
            btn.setAttribute("aria-pressed", "false");
            label.textContent = "Dictate";
            markListening(false);
            box.removeEventListener("keydown", onUserTyping);

            if (gotSpeech) {
                commit();
                if (!errored) {
                    setStatus("Captured. Edit the text above if needed, " +
                              "then analyze.");
                }
            } else if (!errored) {
                setStatus("Did not catch anything. Try again.");
            }
        };

        try {
            rec.start();
        } catch (e) {
            setStatus("Could not start the microphone.", true);
        }
    }

    btn.addEventListener("click", function () {
        if (listening) { stop(); } else { start(); }
    });

    window.addEventListener("pagehide", stop);
})();
</script>
</body>
</html>
"""


def voice_widget_html(palette: dict) -> str:
    """Fill the voice widget template with the active theme colours."""
    variables = "\n    ".join(
        [
            f"--text: {palette['text']};",
            f"--muted: {palette['muted']};",
            f"--border: {palette['border']};",
            f"--strong: {palette['strong']};",
            f"--surface: {palette['surface']};",
            f"--hover: {palette['hover']};",
            f"--accent-text: {palette['accent_text']};",
            f"--neg: {palette['neg']};",
        ]
    )

    return (
        VOICE_TEMPLATE
        .replace("__SCHEME__", palette["scheme"])
        .replace("__VARS__", variables)
        .replace("__LANG__", SPEECH_LANG)
    )


def mount_voice_input():
    """Mount the voice control (st.iframe on new Streamlit, else components)."""
    html = voice_widget_html(palette)

    if hasattr(st, "iframe"):
        st.iframe(html, height=42)
    else:
        components.html(html, height=42)


# ============================================================
# Helpers
# ============================================================

def clean_html(html: str) -> str:
    """Strip indentation and blank lines so Markdown never sees code blocks."""
    return "\n".join(
        line.strip()
        for line in html.splitlines()
        if line.strip()
    )


def render(html: str):
    """Render a block of our own markup, wrapped in the .ss scope."""
    st.html(f'<div class="ss">{clean_html(html)}</div>')


def render_css(css: str):
    """Inject a <style> block (no wrapper)."""
    st.html(clean_html(css))


def esc(value) -> str:
    """Escape dynamic text before putting it into HTML."""
    return htmllib.escape(str(value))


def detect_theme() -> str:
    """
    Follow the system/Streamlit theme on first load. st.context.theme
    needs Streamlit 1.46+; on older versions fall back to dark.
    """
    try:
        detected = st.context.theme.type
        if detected in ("light", "dark"):
            return detected
    except Exception:
        pass

    return "dark"


def toggle_theme():
    st.session_state["theme"] = (
        "light" if st.session_state["theme"] == "dark" else "dark"
    )


def clear_all():
    st.session_state["symptom_text"] = ""
    st.session_state.pop("result", None)
    st.session_state.pop("analyzed_text", None)
    st.session_state.pop("notice", None)


def confidence_level(percent: float):
    """Map a confidence percentage to (key, label, note)."""
    for minimum, key, label, note in CONFIDENCE_LEVELS:
        if percent >= minimum:
            return key, label, note

    _, key, label, note = CONFIDENCE_LEVELS[-1]
    return key, label, note


def term_rows(items, kind: str) -> str:
    """HTML rows for one Integrated Gradients column ('pos' or 'neg')."""

    if not items:
        return '<div class="ss-empty">No terms to show.</div>'

    max_score = max(abs(score) for _, score in items) or 1

    rows = ""

    for word, score in items:

        width = min(100, abs(score) / max_score * 100)

        rows += f"""
        <div class="ss-term">
            <div class="ss-term-word">{esc(word)}</div>
            <div class="ss-bar">
                <i class="{kind}" style="width:{width:.1f}%;"></i>
            </div>
            <div class="ss-num {kind}">{score:+.3f}</div>
        </div>
        """

    return rows


def annotate(text: str, positive, negative):
    """
    Return (html, match_count): the user's own text with the words the
    model relied on highlighted. Only whole-word matches are marked.
    """

    pos_max = max((abs(s) for _, s in positive), default=1) or 1
    neg_max = max((abs(s) for _, s in negative), default=1) or 1

    weights = {}

    # negative first so a word present in both lists shows as supporting
    for word, score in negative:
        key = str(word).replace("##", "").strip().lower()
        if key:
            weights[key] = ("neg", abs(score) / neg_max)

    for word, score in positive:
        key = str(word).replace("##", "").strip().lower()
        if key:
            weights[key] = ("pos", abs(score) / pos_max)

    pieces = []
    matches = 0

    for part in re.split(r"(\w+)", text):

        info = weights.get(part.lower())

        if info:
            kind, relative = info
            alpha = 0.14 + 0.36 * relative
            pieces.append(
                f'<span class="ss-hl {kind}" style="--a:{alpha:.2f};">'
                f'{esc(part)}</span>'
            )
            matches += 1
        else:
            pieces.append(esc(part))

    return "".join(pieces).replace("\n", "<br>"), matches


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="SymptomSense",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "theme" not in st.session_state:
    st.session_state["theme"] = detect_theme()

theme = st.session_state["theme"]
palette = THEMES[theme]


# ============================================================
# Styling
# ============================================================

render_css(
    "<style>@import url('https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500&"
    "family=IBM+Plex+Sans:wght@400;500;600&display=swap');</style>"
)

render_css(
    f"""
    <style>
    :root {{
        color-scheme: {palette["scheme"]};

        --bg: {palette["bg"]};
        --surface: {palette["surface"]};
        --field: {palette["field"]};
        --hover: {palette["hover"]};
        --text: {palette["text"]};
        --muted: {palette["muted"]};
        --faint: {palette["faint"]};
        --border: {palette["border"]};
        --strong: {palette["strong"]};
        --track: {palette["track"]};
        --accent: {palette["accent"]};
        --accent-hover: {palette["accent_hover"]};
        --accent-text: {palette["accent_text"]};
        --on-accent: {palette["on_accent"]};
        --pos: {palette["pos"]};
        --neg: {palette["neg"]};
        --warn: {palette["warn"]};

        --icon-pulse: {ICON_PULSE};
        --icon-info: {ICON_INFO};

        --font: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system,
                "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo,
                Consolas, monospace;
    }}
    </style>
    """
)

render_css(
    """
    <style>

    /* ========================================================
       PAGE + STREAMLIT CHROME
       ======================================================== */

    html,
    body,
    .stApp,
    [data-testid="stApp"] {
        background: var(--bg) !important;
    }

    .stApp {
        color: var(--text);
        font-family: var(--font);
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    .stAppDeployButton,
    #MainMenu,
    footer {
        display: none !important;
    }

    div.block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 920px !important;
        padding: 3.4rem 1.5rem 4rem 1.5rem !important;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.75rem;
    }

    /* Elements that only carry a style block should not add a gap. */
    [data-testid="stElementContainer"]:has(style),
    .element-container:has(style) {
        display: none !important;
    }

    .stApp button,
    .stApp textarea,
    .stApp input,
    .stApp label,
    .stApp p,
    .stApp li {
        font-family: var(--font);
    }

    /* ========================================================
       NATIVE WIDGETS
       ======================================================== */

    /* ---- Buttons ---- */

    .stApp [data-testid="stBaseButton-secondary"],
    .stApp button[kind="secondary"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        min-height: 2.6rem;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: background 0.12s ease, border-color 0.12s ease;
    }

    .stApp [data-testid="stBaseButton-secondary"]:hover,
    .stApp button[kind="secondary"]:hover {
        background: var(--hover) !important;
        border-color: var(--strong) !important;
    }

    .stApp [data-testid="stBaseButton-primary"],
    .stApp button[kind="primary"] {
        background: var(--accent) !important;
        color: var(--on-accent) !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        min-height: 2.6rem;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: background 0.12s ease;
    }

    .stApp [data-testid="stBaseButton-primary"]:hover,
    .stApp button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }

    .stApp button:focus-visible {
        outline: 2px solid var(--accent-text) !important;
        outline-offset: 2px !important;
    }

    .stApp button p,
    .stApp button span,
    .stApp button div {
        color: inherit !important;
    }

    /* theme switch, top right */
    .st-key-theme_btn {
        width: 100% !important;
        display: flex;
        justify-content: flex-end;
    }

    .st-key-theme_btn [data-testid="stButton"],
    .st-key-theme_btn .stButton {
        display: flex;
        justify-content: flex-end;
        width: auto !important;
    }

    .st-key-theme_btn button {
        min-height: 2.2rem !important;
        padding: 0.25rem 0.8rem !important;
    }

    .st-key-theme_btn button p {
        font-size: 0.82rem !important;
    }

    /* ---- Text area ---- */

    .stApp [data-testid="stTextAreaRootElement"],
    .stApp div[data-baseweb="textarea"] {
        background: var(--field) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    .stApp div[data-baseweb="base-input"] {
        background: transparent !important;
        border: 0 !important;
    }

    .stApp [data-testid="stTextAreaRootElement"]:focus-within,
    .stApp div[data-baseweb="textarea"]:focus-within {
        border-color: var(--accent-text) !important;
        box-shadow: 0 0 0 3px rgba(var(--pos), 0.16) !important;
    }

    .stApp [data-testid="stTextAreaRootElement"][data-listening="true"],
    .stApp div[data-baseweb="textarea"][data-listening="true"] {
        border-color: var(--accent-text) !important;
        box-shadow: 0 0 0 3px rgba(var(--pos), 0.16) !important;
    }

    .stApp textarea {
        background: transparent !important;
        color: var(--text) !important;
        caret-color: var(--text);
        font-size: 1rem !important;
        line-height: 1.6 !important;
        padding: 0.85rem 1rem !important;
    }

    .stApp textarea::placeholder {
        color: var(--faint) !important;
        opacity: 1 !important;
    }

    /* ---- Selectbox (control + dropdown, old and new DOM) ---- */

    .stApp [data-testid="stSelectbox"] [role="group"],
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: var(--field) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        min-height: 2.6rem;
        box-shadow: none !important;
    }

    .stApp [data-testid="stSelectbox"] [role="group"]:focus-within,
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
        border-color: var(--accent-text) !important;
        box-shadow: 0 0 0 3px rgba(var(--pos), 0.16) !important;
    }

    .stApp [data-testid="stSelectbox"] input,
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] div[value] {
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        font-size: 0.9rem !important;
    }

    .stApp [data-testid="stSelectbox"] svg {
        color: var(--muted) !important;
        fill: var(--muted) !important;
    }

    /* the dropdown lives in a portal outside .stApp */
    [data-testid="stSelectboxVirtualDropdown"],
    [data-baseweb="popover"] > div,
    [role="listbox"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12) !important;
    }

    [role="option"] {
        background: transparent !important;
        color: var(--text) !important;
        font-size: 0.9rem !important;
    }

    [role="option"] *,
    [role="option"] div {
        color: inherit !important;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [role="option"][data-focused="true"],
    [role="option"][data-focus-visible="true"] {
        background: var(--hover) !important;
    }

    /* ---- Toggle (old and new DOM): track is the label's first div ---- */

    .stApp [data-testid="stCheckbox"] > label > div:first-of-type {
        background-color: var(--strong) !important;
    }

    .stApp [data-testid="stCheckbox"] > label:has(input:checked) > div:first-of-type {
        background-color: var(--accent) !important;
    }

    /* ---- Labels + tooltips ---- */

    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCheckbox"] p {
        color: var(--text) !important;
        font-size: 0.9rem !important;
    }

    .stApp [data-testid="stTooltipIcon"] svg {
        color: var(--faint) !important;
    }

    .stApp [data-testid="stSpinner"],
    .stApp [data-testid="stSpinner"] * {
        color: var(--muted) !important;
    }

    /* ---- Tabs (old and new Streamlit DOM) ---- */

    .stApp [role="tablist"],
    .stApp [data-baseweb="tab-list"] {
        gap: 1.6rem !important;
        border-bottom: 1px solid var(--border) !important;
    }

    .stApp [data-baseweb="tab-border"] {
        background: transparent !important;
    }

    .stApp [role="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        padding: 0.7rem 0 !important;
        margin: 0 !important;
    }

    .stApp [role="tab"] p,
    .stApp [role="tab"] div {
        color: inherit !important;
        font-size: 0.93rem !important;
        font-weight: 500 !important;
    }

    .stApp [role="tab"]:hover,
    .stApp [role="tab"][aria-selected="true"] {
        color: var(--text) !important;
    }

    .stApp [data-baseweb="tab-highlight"],
    .stApp .react-aria-SelectionIndicator {
        background: var(--accent-text) !important;
        height: 2px !important;
        border-radius: 0 !important;
    }

    /* ========================================================
       OUR MARKUP
       ======================================================== */

    .ss {
        font-family: var(--font);
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .ss * {
        box-sizing: border-box;
        font-family: inherit;
    }

    .ss .ss-num,
    .ss .ss-idx,
    .ss .ss-mono {
        font-family: var(--mono) !important;
        font-variant-numeric: tabular-nums;
    }

    .ss-kicker {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
    }

    /* ---- Top bar ---- */

    .ss-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .ss-mark {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        border: 1px solid var(--strong);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--accent-text);
        flex: none;
    }

    .ss-ico {
        display: inline-block;
        width: 18px;
        height: 18px;
        background: currentColor;
        flex: none;
        -webkit-mask-repeat: no-repeat;
        mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
        -webkit-mask-size: contain;
        mask-size: contain;
    }

    .ss-ico-pulse {
        -webkit-mask-image: var(--icon-pulse);
        mask-image: var(--icon-pulse);
        width: 20px;
        height: 20px;
    }

    .ss-ico-info {
        -webkit-mask-image: var(--icon-info);
        mask-image: var(--icon-info);
    }

    .ss-brand-name {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .ss-brand-sub {
        font-size: 0.76rem;
        color: var(--muted);
        line-height: 1.3;
    }

    .ss-rule {
        height: 1px;
        background: var(--border);
    }

    /* ---- Intro ---- */

    .ss-h1 {
        font-size: clamp(1.75rem, 3.4vw, 2.2rem);
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 1.6rem 0 0.6rem 0;
    }

    .ss-lede {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 58ch;
    }

    .ss-notice {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: rgb(var(--neg));
        font-size: 0.88rem;
    }

    .ss-notice .ss-ico {
        width: 16px;
        height: 16px;
    }

    /* ---- Empty-state steps ---- */

    .ss-steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin-top: 2.2rem;
    }

    .ss-step {
        border-top: 1px solid var(--strong);
        padding-top: 0.9rem;
    }

    .ss-step-n {
        font-size: 0.75rem;
        color: var(--accent-text);
    }

    .ss-step-t {
        font-weight: 600;
        margin: 0.3rem 0 0.3rem 0;
    }

    .ss-step-d {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .ss-space {
        height: 1.1rem;
    }

    /* ---- Result ---- */

    .ss-result {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2rem;
        align-items: end;
        padding: 1.6rem 1.7rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface);
    }

    .ss-condition {
        font-size: clamp(1.7rem, 3.4vw, 2.3rem);
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: 1.12;
        margin: 0.4rem 0 0.65rem 0;
        overflow-wrap: anywhere;
    }

    .ss-note {
        color: var(--muted);
        font-size: 0.9rem;
        max-width: 46ch;
    }

    .ss-side {
        min-width: 190px;
    }

    .ss-conf {
        font-family: var(--mono);
        font-size: 2.4rem;
        font-weight: 500;
        letter-spacing: -0.04em;
        line-height: 1;
        margin: 0.45rem 0 0.8rem 0;
    }

    .ss-conf small {
        font-size: 1.1rem;
        color: var(--muted);
        margin-left: 2px;
    }

    .ss-meter {
        height: 4px;
        background: var(--track);
        border-radius: 2px;
        overflow: hidden;
    }

    .ss-meter i {
        display: block;
        height: 100%;
        border-radius: 2px;
        background: var(--level);
    }

    .ss-level {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 0.65rem;
    }

    .ss-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--level);
    }

    .lv-high {
        --level: rgb(var(--pos));
    }

    .lv-moderate {
        --level: rgb(var(--warn));
    }

    .lv-low {
        --level: rgb(var(--neg));
    }

    /* ---- Top predictions ---- */

    .ss-block-head {
        display: flex;
        justify-content: space-between;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    .ss-pred {
        display: grid;
        grid-template-columns: 2rem minmax(0, 1.5fr) minmax(0, 2.5fr) 4.6rem;
        gap: 1rem;
        align-items: center;
        padding: 0.85rem 0;
        border-bottom: 1px solid var(--border);
    }

    .ss-idx {
        color: var(--faint);
        font-size: 0.78rem;
    }

    .ss-pred-name {
        font-weight: 500;
        overflow-wrap: anywhere;
    }

    .ss-pred.first .ss-pred-name {
        font-weight: 600;
    }

    .ss-bar {
        height: 4px;
        background: var(--track);
        border-radius: 2px;
        overflow: hidden;
    }

    .ss-bar i {
        display: block;
        height: 100%;
        border-radius: 2px;
        background: var(--accent);
    }

    .ss-bar i.pos {
        background: rgb(var(--pos));
    }

    .ss-bar i.neg {
        background: rgb(var(--neg));
    }

    .ss-num {
        font-size: 0.85rem;
        text-align: right;
        color: var(--muted);
    }

    .ss-pred.first .ss-num {
        color: var(--text);
    }

    .ss-num.pos {
        color: rgb(var(--pos));
    }

    .ss-num.neg {
        color: rgb(var(--neg));
    }

    /* ---- Explanation ---- */

    .ss-p {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 66ch;
    }

    .ss-card {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
        padding: 1.2rem 1.4rem;
    }

    .ss-annot {
        font-size: 1.05rem;
        line-height: 2.05;
        margin-top: 0.6rem;
    }

    .ss-hl {
        padding: 0.05em 0.25em;
        border-radius: 4px;
        background: rgba(var(--c), var(--a));
        box-shadow: inset 0 -2px 0 rgba(var(--c), 0.9);
    }

    .ss-hl.pos {
        --c: var(--pos);
    }

    .ss-hl.neg {
        --c: var(--neg);
    }

    .ss-legend {
        display: flex;
        gap: 1.4rem;
        flex-wrap: wrap;
        margin-top: 1rem;
        color: var(--muted);
        font-size: 0.78rem;
    }

    .ss-legend span {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }

    .ss-legend i {
        width: 10px;
        height: 10px;
        border-radius: 3px;
    }

    .ss-legend i.pos {
        background: rgb(var(--pos));
    }

    .ss-legend i.neg {
        background: rgb(var(--neg));
    }

    .ss-two {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem 2.6rem;
        margin-top: 1.6rem;
    }

    .ss-list-head {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    .ss-list-head i {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }

    .ss-list-head i.pos {
        background: rgb(var(--pos));
    }

    .ss-list-head i.neg {
        background: rgb(var(--neg));
    }

    .ss-term {
        display: grid;
        grid-template-columns: minmax(80px, 1.1fr) 2fr 3.6rem;
        gap: 0.8rem;
        align-items: center;
        padding: 0.7rem 0;
        border-bottom: 1px solid var(--border);
    }

    .ss-term-word {
        font-weight: 500;
        overflow-wrap: anywhere;
    }

    .ss-empty {
        color: var(--muted);
        font-size: 0.85rem;
        padding: 0.7rem 0;
    }

    .ss-foot {
        color: var(--faint);
        font-size: 0.74rem;
        margin-top: 1.2rem;
    }

    /* ---- Similar cases ---- */

    .ss-case {
        padding: 1.05rem 0;
        border-bottom: 1px solid var(--border);
    }

    .ss-case-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
    }

    .ss-case-title {
        font-weight: 600;
    }

    .ss-case-title .ss-idx {
        margin-right: 0.6rem;
    }

    .ss-case-text {
        color: var(--muted);
        margin-top: 0.35rem;
        line-height: 1.6;
        font-size: 0.92rem;
    }

    /* ---- Condition ---- */

    .ss-cond-text {
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1.2rem;
    }

    .ss-chips {
        margin-top: 0.7rem;
    }

    .ss-chip {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        margin: 0 0.4rem 0.45rem 0;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--bg);
        font-size: 0.82rem;
    }

    /* ---- Disclaimer + footer ---- */

    .ss-disclaimer {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        padding: 0.95rem 1.1rem;
        border: 1px solid rgba(var(--warn), 0.4);
        background: rgba(var(--warn), 0.08);
        border-radius: 10px;
        color: var(--text);
        font-size: 0.84rem;
        line-height: 1.6;
    }

    .ss-disclaimer .ss-ico {
        margin-top: 2px;
        color: rgb(var(--warn));
    }

    .ss-footer {
        color: var(--faint);
        font-size: 0.72rem;
        text-align: center;
    }

    @media (max-width: 720px) {

        .ss-result {
            grid-template-columns: 1fr;
            align-items: start;
        }

        .ss-steps,
        .ss-two {
            grid-template-columns: 1fr;
        }

        .ss-pred {
            grid-template-columns: 1.6rem 1fr 4.2rem;
        }

        .ss-pred .ss-num {
            grid-column: 3;
            grid-row: 1;
        }

        .ss-pred .ss-bar {
            grid-column: 1 / -1;
            grid-row: 2;
        }
    }

    </style>
    """
)


# ============================================================
# Cached inference
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=20
)
def run_inference(
    text: str,
    top_k: int,
    xai_steps: int
):
    return predict_case(
        text,
        top_k=top_k,
        xai_steps=xai_steps
    )


# ============================================================
# Top bar
# ============================================================

try:
    nav_left, nav_right = st.columns(
        [4, 1.3],
        vertical_alignment="center"
    )
except TypeError:
    nav_left, nav_right = st.columns([4, 1.3])

with nav_left:

    render(
        f"""
        <div class="ss-brand">
            <div class="ss-mark">{MARK_SVG}</div>
            <div>
                <div class="ss-brand-name">SymptomSense</div>
                <div class="ss-brand-sub">
                    Clinical decision support prototype
                </div>
            </div>
        </div>
        """
    )

with nav_right:

    theme_label = "Light" if theme == "dark" else "Dark"
    theme_icon = (
        ":material/light_mode:" if theme == "dark"
        else ":material/dark_mode:"
    )

    try:
        st.button(
            theme_label,
            key="theme_btn",
            icon=theme_icon,
            on_click=toggle_theme,
            help=f"Switch to {theme_label.lower()} theme"
        )
    except TypeError:
        st.button(
            theme_label,
            key="theme_btn",
            on_click=toggle_theme
        )

render('<div class="ss-rule"></div>')


# ============================================================
# Intro
# ============================================================

render(
    """
    <div class="ss-h1">Describe your symptoms</div>

    <div class="ss-lede">
        Write in plain language. The model suggests likely conditions,
        shows which of your words drove its answer, and finds similar
        historical cases.
    </div>
    """
)


# ============================================================
# Input
# ============================================================

symptom_text = st.text_area(
    "Symptom description",
    key="symptom_text",
    height=150,
    placeholder=(
        "For example: I've been having a persistent cough, "
        "chest tightness, and difficulty breathing..."
    ),
    label_visibility="collapsed"
)

mount_voice_input()

# ============================================================
# Controls
# ============================================================

control_spec = [2.3, 0.9, 1.7, 1.6]

try:
    control_analyze, control_clear, control_detail, control_cases = (
        st.columns(control_spec, vertical_alignment="center")
    )
except TypeError:
    control_analyze, control_clear, control_detail, control_cases = (
        st.columns(control_spec)
    )

with control_analyze:

    analyze = st.button(
        "Analyze symptoms  →",
        key="analyze_btn",
        type="primary",
        use_container_width=True
    )

with control_clear:

    st.button(
        "Clear",
        key="clear_btn",
        on_click=clear_all,
        use_container_width=True
    )

with control_detail:

    detailed_xai = st.toggle(
        "Detailed explanation",
        value=False,
        help=(
            "Normal mode uses 64 integration steps. "
            "Detailed mode uses 256."
        )
    )

with control_cases:

    top_k = st.selectbox(
        "Similar cases",
        [3, 5],
        index=1,
        format_func=lambda n: f"Show {n} similar cases",
        label_visibility="collapsed"
    )


# ============================================================
# Run analysis
#
# The result lives in session_state so it does not vanish when
# the toggle, dropdown or theme is changed afterwards.
# ============================================================

if analyze:

    if not symptom_text.strip():

        st.session_state.pop("result", None)
        st.session_state.pop("analyzed_text", None)

        render(
            f"""
            <div class="ss-notice">
                {INFO_SVG}
                Enter a symptom description first.
            </div>
            """
        )

    else:

        xai_steps = 256 if detailed_xai else 64

        with st.spinner("Analyzing the symptom description..."):

            st.session_state["result"] = run_inference(
                symptom_text.strip(),
                top_k=top_k,
                xai_steps=xai_steps
            )

            st.session_state["xai_steps"] = xai_steps
            st.session_state["analyzed_text"] = symptom_text.strip()


result = st.session_state.get("result")


# ============================================================
# Empty state
# ============================================================

if not result:

    render(
        """
        <div class="ss-steps">

            <div class="ss-step">
                <div class="ss-step-n ss-mono">01</div>
                <div class="ss-step-t">Classify</div>
                <div class="ss-step-d">
                    BioBERT scores your description against
                    24 conditions.
                </div>
            </div>

            <div class="ss-step">
                <div class="ss-step-n ss-mono">02</div>
                <div class="ss-step-t">Explain</div>
                <div class="ss-step-d">
                    Integrated Gradients shows which words pushed the
                    prediction up or down.
                </div>
            </div>

            <div class="ss-step">
                <div class="ss-step-n ss-mono">03</div>
                <div class="ss-step-t">Compare</div>
                <div class="ss-step-d">
                    Sentence-BERT retrieves similar descriptions from
                    the training data.
                </div>
            </div>

        </div>
        """
    )


# ============================================================
# Results
# ============================================================

if result:

    xai_steps = st.session_state.get("xai_steps", 64)
    analyzed_text = st.session_state.get("analyzed_text", "")

    render('<div class="ss-space"></div><div class="ss-rule"></div>')

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    confidence = float(result["confidence_percent"])
    meter_width = min(100, max(0, confidence))

    level_key, level_label, level_note = confidence_level(confidence)

    render(
        f"""
        <div class="ss-result lv-{level_key}">

            <div>
                <div class="ss-kicker">Most likely condition</div>

                <div class="ss-condition">
                    {esc(result["prediction"])}
                </div>

                <div class="ss-note">{esc(level_note)}</div>
            </div>

            <div class="ss-side">
                <div class="ss-kicker">Confidence</div>

                <div class="ss-conf">{confidence:.1f}<small>%</small></div>

                <div class="ss-meter">
                    <i style="width:{meter_width:.1f}%;"></i>
                </div>

                <div class="ss-level">
                    <span class="ss-dot"></span>
                    {level_label} confidence
                </div>
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # Top predictions
    # --------------------------------------------------------

    rows = ""

    for rank, item in enumerate(result["top_predictions"], start=1):

        percent = item["probability_percent"]
        bar_width = min(100, max(0, percent))
        first = " first" if rank == 1 else ""

        rows += f"""
        <div class="ss-pred{first}">
            <div class="ss-idx">{rank:02d}</div>
            <div class="ss-pred-name">{esc(item["disease"])}</div>
            <div class="ss-bar">
                <i style="width:{bar_width:.2f}%;"></i>
            </div>
            <div class="ss-num">{percent:.2f}%</div>
        </div>
        """

    render(
        f"""
        <div class="ss-space"></div>

        <div class="ss-block-head">
            <span class="ss-kicker">Top predictions</span>
            <span class="ss-kicker">Probability</span>
        </div>
        {rows}
        """
    )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    render('<div class="ss-space"></div>')

    tab_explain, tab_cases, tab_condition = st.tabs(
        [
            "Explanation",
            "Similar cases",
            "Condition"
        ]
    )

    # ---------------- Explanation ----------------

    with tab_explain:

        positive = result["explanation"]["top_positive"][:5]
        negative = result["explanation"]["top_negative"][:5]

        annotated_html, match_count = annotate(
            analyzed_text,
            positive,
            negative
        )

        intro = """
            <div class="ss-p">
                These terms had the strongest influence on the selected
                prediction, according to Integrated Gradients.
            </div>
        """

        annotated_card = ""

        if analyzed_text and match_count:

            annotated_card = f"""
            <div class="ss-card" style="margin-top:1rem;">

                <div class="ss-kicker">Your description</div>

                <div class="ss-annot">{annotated_html}</div>

                <div class="ss-legend">
                    <span><i class="pos"></i>Supports the prediction</span>
                    <span><i class="neg"></i>Opposes the prediction</span>
                </div>

            </div>
            """

        delta = result["explanation"]["relative_delta_percent"]

        render(
            f"""
            {intro}
            {annotated_card}

            <div class="ss-two">

                <div>
                    <div class="ss-list-head">
                        <i class="pos"></i>
                        <span class="ss-kicker">Supporting terms</span>
                    </div>
                    {term_rows(positive, "pos")}
                </div>

                <div>
                    <div class="ss-list-head">
                        <i class="neg"></i>
                        <span class="ss-kicker">Opposing terms</span>
                    </div>
                    {term_rows(negative, "neg")}
                </div>

            </div>

            <div class="ss-foot ss-mono">
                Integrated Gradients · {xai_steps} steps ·
                convergence error {delta:.2f}%
            </div>
            """
        )

    # ---------------- Similar cases ----------------

    with tab_cases:

        cases_html = ""

        for i, case in enumerate(
            result["similar_cases"],
            start=1
        ):

            similarity = case["similarity"] * 100

            cases_html += f"""
            <div class="ss-case">

                <div class="ss-case-head">

                    <div class="ss-case-title">
                        <span class="ss-idx">{i:02d}</span>{esc(case["disease_label"])}
                    </div>

                    <div class="ss-num">{similarity:.1f}% similar</div>

                </div>

                <div class="ss-case-text">
                    {esc(case["symptom_text"])}
                </div>

            </div>
            """

        render(
            f"""
            <div class="ss-p">
                Examples from the training corpus that are semantically
                similar to the symptom description.
            </div>

            <div style="margin-top:0.6rem;">{cases_html}</div>
            """
        )

    # ---------------- Condition ----------------

    with tab_condition:

        info = result["disease_information"]

        description = info.get(
            "description",
            "No description available."
        )

        symptoms = info.get("common_symptoms", [])

        source = info.get("source", "Not specified")

        chips = "".join(
            f'<span class="ss-chip">{esc(symptom)}</span>'
            for symptom in symptoms
        )

        symptom_block = ""

        if chips:

            symptom_block = f"""
            <div class="ss-kicker">Commonly associated symptoms</div>
            <div class="ss-chips">{chips}</div>
            """

        render(
            f"""
            <div class="ss-card" style="margin-top:0.8rem;">

                <div class="ss-cond-text">{esc(description)}</div>

                {symptom_block}

                <div class="ss-foot">Source: {esc(source)}</div>

            </div>
            """
        )

    # --------------------------------------------------------
    # Disclaimer + footer
    # --------------------------------------------------------

    render(
        f"""
        <div class="ss-disclaimer">
            {INFO_SVG}
            <div>
                <strong>Not a diagnosis.</strong>
                This prototype is intended for preliminary screening and
                educational use. Its output is a model prediction and
                should not replace evaluation by a qualified healthcare
                professional.
            </div>
        </div>

        <div class="ss-footer ss-mono" style="margin-top:1.4rem;">
            SymptomSense · BioBERT · Integrated Gradients · Sentence-BERT
        </div>
        """
    )