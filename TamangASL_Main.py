import flet as ft
import cv2
import base64
import threading
import time
import collections
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#F0F4FA"
SURFACE   = "#FFFFFF"
SURFACE2  = "#E8EEF7"
BORDER    = "#C5D1E8"
ACCENT    = "#0D2461"
ACCENT2   = "#1A3580"
GOLD      = "#1A3580"
TEXT_HI   = "#0A1628"
TEXT_MED  = "#4A5E7A"
TEXT_LOW  = "#8A9DB8"
SUCCESS   = "#2ECC71"
WARNING   = "#F39C12"
ERROR_COL = "#E74C3C"

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS MAPPING
#  image_dataset_from_directory sorts folder names alphabetically.
#  Folder names: "0"–"9" then "a"–"z"  →  indices 0–9 then 10–35.
# ══════════════════════════════════════════════════════════════════════════════
CATEGORIES: dict[int, str] = {
    0:"a", 1:"b", 2:"c", 3:"d", 4:"e",
    5:"f", 6:"g", 7:"h", 8:"i", 9:"j",
    10:"k", 11:"l", 12:"m", 13:"n", 14:"o",
    15:"p", 16:"q", 17:"r", 18:"s", 19:"t",
    20:"u", 21:"v", 22:"w", 23:"x", 24:"y",
    25:"z",
}

# Letters that require motion — cannot be reliably captured as statics
MOTION_SIGNS = {"j", "z"}

# Promptable letters (exclude motion signs, include digits 0-9)
PROMPTABLE = [v for v in CATEGORIES.values() if v not in MOTION_SIGNS]
LETTER_ONLY = [c for c in PROMPTABLE if c.isalpha()]

# Smoothing: keep a rolling window of predictions and pick most common
SMOOTH_WINDOW = 5

# Confidence threshold: predictions below this are treated as uncertain
CONFIDENCE_THRESHOLD = 55.0   # percent


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def chip(icon, label_text, value_ref, page_ref):
    """Small rounded stat chip."""
    return ft.Container(
        content=ft.Row([
            ft.Icon(icon, size=17, color=ACCENT),
            ft.Text(label_text, size=13, color=TEXT_MED),
            value_ref,
        ], spacing=5, tight=True),
        bgcolor=SURFACE2,
        border=ft.border.all(1, BORDER),
        border_radius=20,
        padding=ft.padding.symmetric(horizontal=14, vertical=8),
    )


def section_header(title: str, subtitle: str):
    return ft.Row([
        ft.Container(width=4, height=36, bgcolor=ACCENT, border_radius=2),
        ft.Column([
            ft.Text(title,    size=20, weight=ft.FontWeight.W_700, color=TEXT_HI),
            ft.Text(subtitle, size=13, color=TEXT_MED),
        ], spacing=1, tight=True),
    ], spacing=10)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main(page: ft.Page):
    # ── Page ──────────────────────────────────────────────────────────────────
    page.title         = "Tamang ASL"
    page.bgcolor       = BG
    page.padding       = 0
    page.window_width  = 1440
    page.window_height = 900
    page.window_min_width  = 1100
    page.window_min_height = 750
    page.window_maximized  = True
    # min_height set above
    page.fonts = {
        "Display": "https://fonts.gstatic.com/s/spacegrotesk/v15/V8mQoQDjQSkFtoMM3T6r8E7mF71Q-gozuUa4-LQ.woff2",
        "Mono":    "https://fonts.gstatic.com/s/jetbrainsmono/v18/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPVmUsaaDhw.woff2",
    }
    page.theme = ft.Theme(font_family="Display")

    # ── State ─────────────────────────────────────────────────────────────────
    import random
    _quiz_letters = random.sample(LETTER_ONLY, 10)
    state = {
        "capture_flag": False,
        "is_capturing": False,
        "current_tab":  "practice",
        "current_prompt": _quiz_letters[0],
        "streak":       0,
        "best_streak":  0,
        "attempts":     0,
        "correct_total":0,
        "word_letters": [],
        "pred_buffer":  collections.deque(maxlen=SMOOTH_WINDOW),
        "practice_letter": "a",
        "quiz_letters":  _quiz_letters,   # 10 unique letters for current round
        "quiz_index":    0,               # which letter we're on (0-9)
        "quiz_progress": [None] * 10,     # None=unanswered, True=correct, False=wrong
    }

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # ── TFLite ────────────────────────────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()
    out_d = interpreter.get_output_details()
    _, MODEL_H, MODEL_W, _ = inp_d[0]["shape"]

    # ══════════════════════════════════════════════════════════════════════════
    #  UI COMPONENTS
    # ══════════════════════════════════════════════════════════════════════════

    # Stat labels
    streak_lbl   = ft.Text("0", size=14, color=TEXT_HI, weight=ft.FontWeight.BOLD)
    best_lbl     = ft.Text("0", size=14, color=GOLD,    weight=ft.FontWeight.BOLD)
    accuracy_lbl = ft.Text("—%", size=14, color=TEXT_HI, weight=ft.FontWeight.BOLD)

    # Camera feed
    DUMMY_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAj"
                 "CB0C8AAAAASUVORK5CYII=")
    camera_img = ft.Image(
        src_base64=DUMMY_B64, fit=ft.ImageFit.COVER,
        border_radius=10, expand=True,
    )
    camera_wrap = ft.Container(
        content=camera_img, bgcolor="#E8EEF7",
        border_radius=10, border=ft.border.all(1, BORDER),
        expand=True,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )

    # Hand status pill
    hand_dot   = ft.Container(width=10, height=10, bgcolor=TEXT_LOW, border_radius=4)
    hand_label = ft.Text("No hand", size=13, color=TEXT_MED)
    hand_pill  = ft.Container(
        content=ft.Row([hand_dot, hand_label], spacing=6, tight=True),
        bgcolor="#F0F4FACC", border_radius=20,
        padding=ft.padding.symmetric(horizontal=14, vertical=6),
    )

    # Live confidence bar (shown during preview, before capture)
    conf_bar_track = ft.Container(
        content=ft.Container(
            width=0, height=8, bgcolor=ACCENT, border_radius=4,
        ),
        width=280, height=8, bgcolor=SURFACE2,
        border_radius=3, border=ft.border.all(1, BORDER),
    )
    conf_label = ft.Text("–", size=13, color=TEXT_MED, font_family="Mono")
    conf_row = ft.Row([
        ft.Text("Live:", size=13, color=TEXT_LOW),
        conf_bar_track,
        conf_label,
    ], spacing=8, tight=True)

    # ── Practice Panel Controls ───────────────────────────────────────────────

    prompt_letter = ft.Text(
        state["current_prompt"].upper(), size=140,
        weight=ft.FontWeight.W_900, color=ACCENT,
        text_align=ft.TextAlign.CENTER,
    )
    prompt_sub = ft.Text(
        "Sign this letter", size=14, color=TEXT_MED,
        text_align=ft.TextAlign.CENTER,
    )

    # Animated result badge — Quiz mode
    result_badge = ft.Container(
        visible=False, border_radius=24,
        padding=ft.padding.symmetric(horizontal=14, vertical=8),
        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
    )

    # Animated result badge — Practice (guided) mode
    practice_result_badge = ft.Container(
        visible=False, border_radius=24,
        padding=ft.padding.symmetric(horizontal=14, vertical=8),
        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
    )

    def _fill_badge(badge, correct, letter, confidence, top3, raw_idx):
        display = letter.upper()
        if correct:
            color, icon = SUCCESS, ft.Icons.CHECK_CIRCLE_ROUNDED
            msg = f"Correct! You signed {display}"
            badge.bgcolor = "#E8F5EE"
            badge.border  = ft.border.all(1, "#2ECC71")
        else:
            color, icon = ERROR_COL, ft.Icons.CANCEL_ROUNDED
            msg = f"Got {display} ({confidence:.0f}%). Keep going!"
            badge.bgcolor = "#F5D5D8"
            badge.border  = ft.border.all(1, ERROR_COL)
        badge.content = ft.Row([
            ft.Icon(icon, color=color, size=22),
            ft.Text(msg, color=color, size=14),
        ], spacing=8, tight=True)
        badge.visible = True

    def _update_stats():
        streak_lbl.value   = str(state["streak"])
        best_lbl.value     = str(state["best_streak"])
        acc_pct = (state["correct_total"] / state["attempts"] * 100) if state["attempts"] else 0
        accuracy_lbl.value = f"{acc_pct:.0f}%"

    def show_quiz_result(correct, letter, confidence, top3, raw_idx):
        idx = state["quiz_index"]
        if correct:
            state["correct_total"] += 1
            state["streak"] += 1
            if state["streak"] > state["best_streak"]:
                state["best_streak"] = state["streak"]
            if state["quiz_progress"][idx] is not False:
                state["quiz_progress"][idx] = True
            next_idx = idx + 1
            state["quiz_index"] = next_idx
            quiz_done = next_idx >= DOTS
            cap_btn.visible     = False
            skip_btn.visible    = False
            next_btn.visible    = not quiz_done
            restart_btn.visible = quiz_done
        else:
            state["streak"] = 0
            state["quiz_progress"][idx] = False
            cap_btn.visible     = True
            skip_btn.visible    = True   # show skip after first wrong
            next_btn.visible    = False
            restart_btn.visible = False
        _update_stats()
        refresh_dots()
        _fill_badge(result_badge, correct, letter, confidence, top3, raw_idx)
        page.update()

    def show_practice_result(correct, letter, confidence, top3, raw_idx):
        if correct:
            state["correct_total"] += 1
            state["streak"] += 1
            if state["streak"] > state["best_streak"]:
                state["best_streak"] = state["streak"]
        else:
            state["streak"] = 0
        _update_stats()
        _fill_badge(practice_result_badge, correct, letter, confidence, top3, raw_idx)
        page.update()

    # ── Word Panel Controls ───────────────────────────────────────────────────
    word_display = ft.Text(
        "", size=42, color=TEXT_HI, font_family="Mono",
        text_align=ft.TextAlign.CENTER,
    )
    word_hint = ft.Text(
        "Spell a word letter-by-letter", size=14, color=TEXT_MED,
        text_align=ft.TextAlign.CENTER,
    )

    def on_clear_word(_):
        state["word_letters"] = []
        word_display.value = ""
        word_hint.value    = "Spell a word letter-by-letter"
        word_search_btn.visible = False
        page.update()

    def on_backspace_word(_):
        if state["word_letters"]:
            state["word_letters"].pop()
            word_display.value = " ".join(l.upper() for l in state["word_letters"])
            word_hint.value    = f"{len(state['word_letters'])} letter(s)"
        word_search_btn.visible = len(state["word_letters"]) > 0
        page.update()

    def on_search_word(_):
        word = "".join(state["word_letters"])
        if word:
            import webbrowser
            webbrowser.open(f"https://www.signasl.org/sign/{word}")

    word_search_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.SEARCH_ROUNDED, color=ACCENT, size=18),
            ft.Text('Search on web', size=14, color=ACCENT),
        ], spacing=6, tight=True),
        border=ft.border.all(1, BORDER),
        border_radius=40,
        padding=ft.padding.symmetric(horizontal=16, vertical=10),
        on_click=on_search_word, ink=True,
        visible=False,
    )

    # ── Capture Button ─────────────────────────────────────────────────────────
    btn_text = ft.Text("Capture Sign", size=15, color="#FFFFFF", weight=ft.FontWeight.W_600)
    cap_row  = ft.Row([
        ft.Icon(ft.Icons.FRONT_HAND_OUTLINED, color="#FFFFFF", size=20),
        btn_text,
    ], spacing=8, tight=True)

    def on_capture(_):
        if state["is_capturing"]:
            return
        state["is_capturing"] = True
        state["capture_flag"] = True
        btn_text.value = "Processing…"
        prac_btn_text.value = "Processing…"
        state["pred_buffer"].clear()
        page.update()

    cap_btn = ft.Container(
        content=cap_row, bgcolor=ACCENT,
        border_radius=40,
        padding=ft.padding.symmetric(horizontal=32, vertical=16),
        on_click=on_capture, ink=True,
        shadow=ft.BoxShadow(blur_radius=20, color="#0D246144", offset=ft.Offset(0, 5)),
        animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
    )

    # Separate capture button for practice mode — never hidden by quiz logic
    prac_btn_text = ft.Text("Capture Sign", size=15, color="#FFFFFF", weight=ft.FontWeight.W_600)
    prac_cap_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.FRONT_HAND_OUTLINED, color="#FFFFFF", size=20),
            prac_btn_text,
        ], spacing=8, tight=True),
        bgcolor=ACCENT,
        border_radius=40,
        padding=ft.padding.symmetric(horizontal=32, vertical=16),
        on_click=on_capture, ink=True,
        shadow=ft.BoxShadow(blur_radius=20, color="#0D246144", offset=ft.Offset(0, 5)),
        animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
    )

    # ── Next Button ────────────────────────────────────────────────────────────
    def on_next(_):
        idx = state["quiz_index"]
        if idx >= DOTS:
            return
        state["current_prompt"] = state["quiz_letters"][idx]
        prompt_letter.value  = state["current_prompt"].upper()
        result_badge.visible = False
        cap_btn.visible      = True
        next_btn.visible     = False
        skip_btn.visible     = False
        restart_btn.visible  = False
        state["pred_buffer"].clear()
        page.update()

    next_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.ARROW_FORWARD_ROUNDED, color="#FFFFFF", size=20),
            ft.Text("Next", size=15, color="#FFFFFF", weight=ft.FontWeight.W_600),
        ], spacing=8, tight=True),
        bgcolor=ACCENT,
        border_radius=40,
        padding=ft.padding.symmetric(horizontal=32, vertical=16),
        on_click=on_next, ink=True,
        shadow=ft.BoxShadow(blur_radius=20, color="#0D246144", offset=ft.Offset(0, 5)),
        animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
        visible=False,
    )

    def on_restart(_):
        new_letters = random.sample(LETTER_ONLY, 10)
        state["quiz_letters"]  = new_letters
        state["quiz_index"]    = 0
        state["quiz_progress"] = [None] * 10
        state["streak"]        = 0
        state["current_prompt"] = new_letters[0]
        prompt_letter.value   = new_letters[0].upper()
        refresh_dots()
        restart_btn.visible  = False
        result_badge.visible = False
        cap_btn.visible      = True
        next_btn.visible     = False
        skip_btn.visible     = False
        state["pred_buffer"].clear()
        page.update()

    restart_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.REFRESH_ROUNDED, color=WARNING, size=18),
            ft.Text("Restart", size=14, color=WARNING),
        ], spacing=6, tight=True),
        border=ft.border.all(1, WARNING+"66"),
        border_radius=40,
        padding=ft.padding.symmetric(horizontal=28, vertical=16),
        on_click=on_restart, ink=True,
        visible=False,
    )

    def on_skip(_):
        idx = state["quiz_index"]
        if idx >= DOTS:
            return
        # Mark as wrong (red) if not already marked
        if state["quiz_progress"][idx] is None:
            state["quiz_progress"][idx] = False
        # Advance
        next_idx = idx + 1
        state["quiz_index"] = next_idx
        quiz_done = next_idx >= DOTS
        state["current_prompt"] = state["quiz_letters"][next_idx] if not quiz_done else state["quiz_letters"][idx]
        prompt_letter.value  = state["current_prompt"].upper()
        result_badge.visible = False
        cap_btn.visible      = not quiz_done
        skip_btn.visible     = False
        next_btn.visible     = False
        restart_btn.visible  = quiz_done
        refresh_dots()
        state["pred_buffer"].clear()
        page.update()

    skip_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.SKIP_NEXT_ROUNDED, color=TEXT_MED, size=18),
            ft.Text("Skip", size=14, color=TEXT_MED),
        ], spacing=6, tight=True),
        border=ft.border.all(1, BORDER),
        border_radius=40,
        padding=ft.padding.symmetric(horizontal=22, vertical=12),
        on_click=on_skip, ink=True,
        visible=False,
    )

    # ── Progress dots (quiz progress: 10 questions) ───────────────────────────
    DOTS = 10

    dots_row = ft.Row(spacing=6, tight=True,
                      controls=[
                          ft.Container(width=12, height=12,
                                       bgcolor=SURFACE2, border_radius=6,
                                       border=ft.border.all(1, BORDER))
                          for _ in range(DOTS)
                      ])

    def refresh_dots():
        prog = state["quiz_progress"]
        for i, dot in enumerate(dots_row.controls):
            val = prog[i]
            if val is True:
                dot.bgcolor = SUCCESS
                dot.border  = ft.border.all(1, SUCCESS)
            elif val is False:
                dot.bgcolor = ERROR_COL
                dot.border  = ft.border.all(1, ERROR_COL)
            else:  # None — not yet reached
                dot.bgcolor = SURFACE2
                dot.border  = ft.border.all(1, BORDER)

    # ── Practice Mode (guided) components ────────────────────────────────────
    PRACTICE_LETTERS = [c for c in LETTER_ONLY if c.isalpha()]  # a-z minus j,z

    practice_letter_lbl = ft.Text(
        state["practice_letter"].upper(), size=120,
        weight=ft.FontWeight.W_900, color=ACCENT,
        text_align=ft.TextAlign.CENTER,
    )

    practice_hand_img = ft.Image(
        src=f"assets/learn_{state['practice_letter']}.png",
        width=180, height=180, fit=ft.ImageFit.CONTAIN,
        border_radius=12,
    )

    def update_practice_letter(letter: str):
        state["practice_letter"] = letter
        practice_letter_lbl.value = letter.upper()
        practice_hand_img.src = f"assets/learn_{letter}.png"
        # Update button highlights
        for btn in letter_btn_row_controls:
            btn.bgcolor = ACCENT if btn.data == letter else SURFACE2
            btn.border  = ft.border.all(1, ACCENT if btn.data == letter else BORDER)
            btn.content.color = "#FFFFFF" if btn.data == letter else TEXT_MED
        page.update()

    def make_letter_btn(letter: str):
        active = letter == state["practice_letter"]
        return ft.Container(
            data=letter,
            content=ft.Text(
                letter.upper(), size=13, weight=ft.FontWeight.W_600,
                color="#FFFFFF" if active else TEXT_MED,
                text_align=ft.TextAlign.CENTER,
            ),
            bgcolor=ACCENT if active else SURFACE2,
            border=ft.border.all(1, ACCENT if active else BORDER),
            border_radius=8,
            width=34, height=34,
            alignment=ft.alignment.center,
            on_click=lambda e, l=letter: update_practice_letter(l),
            ink=True,
        )

    letter_btn_row_controls = [make_letter_btn(l) for l in PRACTICE_LETTERS]

    # ══════════════════════════════════════════════════════════════════════════
    #  PANELS
    # ══════════════════════════════════════════════════════════════════════════

    def quiz_panel():
        return ft.Column([
            section_header("Quiz Mode", "Sign the random letter shown below"),
            ft.Divider(color=BORDER, height=1),
            ft.Container(
                content=ft.Column([
                    prompt_letter,
                    prompt_sub,
                    ft.Container(height=4),
                    dots_row,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
                bgcolor=SURFACE2,
                border=ft.border.all(1, BORDER),
                border_radius=16,
                padding=ft.padding.symmetric(vertical=20, horizontal=20),
                alignment=ft.alignment.center,
            ),
            ft.Row([result_badge], alignment=ft.MainAxisAlignment.CENTER),
            conf_row,
            ft.Row([cap_btn, skip_btn, next_btn, restart_btn],
                   alignment=ft.MainAxisAlignment.CENTER, spacing=12),
        ], spacing=14, expand=True)

    def practice_panel():
        # Alphabet grid: 12 cols × 2 rows
        rows = []
        row_size = 12
        for i in range(0, len(letter_btn_row_controls), row_size):
            rows.append(
                ft.Row(
                    letter_btn_row_controls[i:i+row_size],
                    spacing=4,
                    tight=True,
                    alignment=ft.MainAxisAlignment.CENTER,
                )
            )

        return ft.Column([
            section_header("Practice Mode", "Select a letter and study the hand sign"),
            ft.Divider(color=BORDER, height=1),
            # Letter selector
            ft.Container(
                content=ft.Column(rows, spacing=4, tight=True,
                                  horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=SURFACE2,
                border=ft.border.all(1, BORDER),
                border_radius=12,
                padding=ft.padding.symmetric(vertical=10, horizontal=10),
                alignment=ft.alignment.center,
            ),
            # Letter + hand image side by side
            ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=practice_letter_lbl,
                        expand=True,
                        alignment=ft.alignment.center,
                    ),
                    ft.Container(
                        content=practice_hand_img,
                        bgcolor=SURFACE,
                        border=ft.border.all(1, BORDER),
                        border_radius=12,
                        padding=8,
                        width=200,
                        height=200,
                        alignment=ft.alignment.center,
                    ),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                bgcolor=SURFACE2,
                border=ft.border.all(1, BORDER),
                border_radius=16,
                padding=ft.padding.symmetric(vertical=12, horizontal=16),
            ),
            conf_row,
            ft.Row([practice_result_badge], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([prac_cap_btn],
                   alignment=ft.MainAxisAlignment.CENTER, spacing=12),
        ], spacing=14, expand=True)

    def word_panel():
        return ft.Column([
            section_header("Word Builder", "Spell a word letter-by-letter"),
            ft.Divider(color=BORDER, height=1),
            ft.Container(
                content=ft.Column([
                    ft.Text("Current Word", size=13, color=TEXT_MED),
                    word_display,
                    word_hint,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
                bgcolor=SURFACE2,
                border=ft.border.all(1, BORDER),
                border_radius=16,
                padding=ft.padding.symmetric(vertical=24, horizontal=20),
                alignment=ft.alignment.center,
            ),
            conf_row,
            ft.Row([
                cap_btn,
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.BACKSPACE_OUTLINED, color=TEXT_MED, size=18),
                        ft.Text("Back", size=14, color=TEXT_MED),
                    ], spacing=6, tight=True),
                    border=ft.border.all(1, BORDER),
                    border_radius=40,
                    padding=ft.padding.symmetric(horizontal=16, vertical=10),
                    on_click=on_backspace_word, ink=True,
                ),
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DELETE_OUTLINE_ROUNDED, color=ERROR_COL, size=18),
                        ft.Text("Clear", size=14, color=ERROR_COL),
                    ], spacing=6, tight=True),
                    border=ft.border.all(1, BORDER),
                    border_radius=40,
                    padding=ft.padding.symmetric(horizontal=16, vertical=10),
                    on_click=on_clear_word, ink=True,
                ),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
            ft.Row([word_search_btn], alignment=ft.MainAxisAlignment.CENTER),
        ], spacing=16, expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB SYSTEM
    # ══════════════════════════════════════════════════════════════════════════

    tab_practice_ref = ft.Ref[ft.Container]()
    tab_quiz_ref     = ft.Ref[ft.Container]()
    tab_word_ref     = ft.Ref[ft.Container]()
    right_panel_ref  = ft.Ref[ft.Column]()

    def make_tab(ref, icon, label, active, on_click):
        c = "#FFFFFF" if active else TEXT_MED
        return ft.Container(
            ref=ref,
            content=ft.Row([
                ft.Icon(icon, color=c, size=16),
                ft.Text(label, size=13, color=c),
            ], spacing=5, tight=True),
            bgcolor=ACCENT if active else "transparent",
            border=ft.border.all(1, ACCENT if active else BORDER),
            border_radius=24,
            padding=ft.padding.symmetric(horizontal=16, vertical=9),
            on_click=on_click, ink=True,
        )

    TAB_PANELS = {
        "practice": practice_panel,
        "quiz":     quiz_panel,
        "word":     word_panel,
    }
    TAB_REFS = {}  # filled after refs are created

    def switch_tab(tab: str):
        state["current_tab"] = tab
        # Hide all badges when switching tabs
        result_badge.visible          = False
        practice_result_badge.visible = False
        ref_map = {
            "practice": tab_practice_ref,
            "quiz":     tab_quiz_ref,
            "word":     tab_word_ref,
        }
        for name, ref in ref_map.items():
            active = name == tab
            ref.current.bgcolor = ACCENT if active else "transparent"
            ref.current.border  = ft.border.all(1, ACCENT if active else BORDER)
            for ctrl in ref.current.content.controls:
                ctrl.color = "#FFFFFF" if active else TEXT_MED
        right_panel_ref.current.controls[-1] = TAB_PANELS[tab]()
        page.update()

    def on_tab_practice(_): switch_tab("practice")
    def on_tab_quiz(_):     switch_tab("quiz")
    def on_tab_word(_):     switch_tab("word")

    # ══════════════════════════════════════════════════════════════════════════
    #  LAYOUT ASSEMBLY
    # ══════════════════════════════════════════════════════════════════════════

    sidebar = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Text("T", size=26, weight=ft.FontWeight.W_900,
                                color=ACCENT),
                width=46, height=46,
                bgcolor="#0D246122",
                border=ft.border.all(1, "#0D246155"),
                border_radius=12,
                alignment=ft.alignment.center,
            ),
            ft.Container(height=8),
            ft.Text("ASL", size=8, color=TEXT_LOW, text_align=ft.TextAlign.CENTER),
            ft.Container(expand=True),
            ft.Text("v1.0", size=8, color=TEXT_LOW),
        ], alignment=ft.MainAxisAlignment.START,
           horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
        width=62,
        bgcolor=SURFACE,
        border=ft.border.all(0, "transparent"),
        padding=ft.padding.symmetric(vertical=18, horizontal=8),
    )

    logo_box = ft.Container(
        content=ft.Stack([
            ft.Container(
                width=56, height=56,
                bgcolor=ACCENT,
                border_radius=13,
            ),
            ft.Container(
                content=ft.Icon(ft.Icons.SIGN_LANGUAGE_ROUNDED, color="#FFFFFF", size=30),
                width=56, height=56,
                alignment=ft.alignment.center,
            ),
            ft.Container(
                content=ft.Container(
                    width=15, height=15,
                    bgcolor="#6B8FC9",
                    border_radius=ft.border_radius.only(top_left=7, bottom_right=7),
                    border=ft.border.all(2, SURFACE),
                ),
                alignment=ft.alignment.bottom_right,
                width=56, height=56,
            ),
        ]),
        width=56, height=56,
    )

    header = ft.Container(
        content=ft.Row([
            ft.Row([
                logo_box,
                ft.Column([
                    ft.Text("Tamang ASL", size=26, weight=ft.FontWeight.W_800,
                            color=ACCENT),
                    ft.Text("American Sign Language Practice", size=13, color=TEXT_MED),
                ], spacing=0, tight=True),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(expand=True),
            chip(ft.Icons.BOLT_ROUNDED,    "Streak",   streak_lbl,   page),
            chip(ft.Icons.EMOJI_EVENTS_ROUNDED, "Best", best_lbl,    page),
            chip(ft.Icons.PERCENT_ROUNDED, "Accuracy", accuracy_lbl, page),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
        bgcolor=SURFACE,
        border=ft.border.all(1, BORDER),
        border_radius=ft.border_radius.only(
            bottom_left=12, bottom_right=12),
        padding=ft.padding.symmetric(horizontal=22, vertical=12),
        margin=ft.margin.only(bottom=14),
    )

    cam_section = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Stack([
                    camera_wrap,
                    ft.Container(content=hand_pill,
                                 alignment=ft.alignment.bottom_left, padding=10),
                ], expand=True),
                expand=True,
                alignment=ft.alignment.center,
            ),
            ft.Container(
                content=ft.Text("Live Camera Feed", size=12, color=TEXT_LOW),
                alignment=ft.alignment.center,
                padding=ft.padding.only(top=4),
            ),
        ], spacing=4, expand=True,
           horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        expand=True,
        alignment=ft.alignment.center,
        bgcolor=BG,
    )

    right_col = ft.Column(
        ref=right_panel_ref,
        controls=[
            ft.Row([
                make_tab(tab_practice_ref, ft.Icons.MENU_BOOK_ROUNDED,
                         "Practice", True,  on_tab_practice),
                make_tab(tab_quiz_ref,     ft.Icons.FRONT_HAND_OUTLINED,
                         "Quiz", False, on_tab_quiz),
                make_tab(tab_word_ref,     ft.Icons.SPELLCHECK_ROUNDED,
                         "Word Builder", False, on_tab_word),
            ], spacing=6, wrap=True),
            practice_panel(),
        ],
        spacing=16, expand=True,
    )

    right_container = ft.Container(
        content=right_col,
        bgcolor=SURFACE,
        border=ft.border.all(1, BORDER),
        border_radius=16,
        padding=24,
        width=560,
    )

    body = ft.Row([
        cam_section,
        right_container,
    ], spacing=20, expand=True,
       vertical_alignment=ft.CrossAxisAlignment.STRETCH)

    root = ft.Container(
        content=ft.Column([header, body], spacing=0, expand=True),
        expand=True,
        padding=ft.padding.only(left=24, right=24, bottom=22),
    )

    page.add(root)

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMERA THREAD
    # ══════════════════════════════════════════════════════════════════════════

    def camera_loop():
        cap = cv2.VideoCapture(0)
        x_min = y_min = x_max = y_max = 0
        hand_found = False
        last_ui_update = 0.0
        UI_FPS = 20          

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. THE MIRROR FIX: Flip the frame so a Right Hand stays a Right Hand!
            frame = cv2.flip(frame, 1)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_res = detector.detect(mp_img)
            h, w, _ = frame.shape

            hand_found = bool(mp_res.hand_landmarks)
            clean_crop = np.array([]) # Initialize an empty safety variable

            # Update hand status
            if hand_found:
                hand_dot.bgcolor   = SUCCESS
                hand_label.value   = "Hand detected"
                hand_label.color   = SUCCESS
            else:
                hand_dot.bgcolor   = TEXT_LOW
                hand_label.value   = "No hand"
                hand_label.color   = TEXT_MED

            # Bounding box
            if hand_found:
                for lms in mp_res.hand_landmarks:
                    xs  = [int(lm.x * w) for lm in lms]
                    ys  = [int(lm.y * h) for lm in lms]
                    
                    pad = 20 
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)
                    
                    x_min -= pad
                    y_min -= pad
                    x_max += pad
                    y_max += pad
                    
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    max_side = max(box_width, box_height)
                    
                    center_x = x_min + (box_width // 2)
                    center_y = y_min + (box_height // 2)
                    
                    x_min = int(center_x - (max_side / 2))
                    y_min = int(center_y - (max_side / 2))
                    x_max = x_min + max_side
                    y_max = y_min + max_side
                    
                    if x_min < 0:
                        x_max += abs(x_min)
                        x_min = 0
                    if y_min < 0:
                        y_max += abs(y_min)
                        y_min = 0
                    if x_max > w:
                        x_min -= (x_max - w)
                        x_max = w
                    if y_max > h:
                        y_min -= (y_max - h)
                        y_max = h
                        
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    # 2. THE GREEN BOX FIX: Grab the crop BEFORE drawing the rectangle!
                    clean_crop = frame[y_min:y_max, x_min:x_max].copy()
                    
                    # NOW draw the green rectangle for the UI
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (0, 229, 204), 2)

                # ── Live soft prediction ──────
                if clean_crop.size > 0:
                    # We pass the clean crop (with no green lines) into the AI
                    rs  = cv2.resize(clean_crop, (MODEL_W, MODEL_H))
                    rgb_c = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
                    inp = np.expand_dims(rgb_c, axis=0).astype(np.float32)
                    interpreter.set_tensor(inp_d[0]["index"], inp)
                    interpreter.invoke()
                    probs = interpreter.get_tensor(out_d[0]["index"])[0]
                    live_idx  = int(np.argmax(probs))
                    live_conf = float(probs[live_idx]) * 100
                    live_let  = CATEGORIES[live_idx]
                    state["pred_buffer"].append(live_idx)

                    bar_w = int(200 * live_conf / 100)
                    conf_bar_track.content.width = bar_w
                    conf_bar_track.content.bgcolor = (
                        ACCENT if live_conf >= CONFIDENCE_THRESHOLD else "#6B8FC9"
                    )
                    conf_label.value = f"{live_let.upper()} {live_conf:.0f}%"
            else:
                conf_label.value = "–"
                conf_bar_track.content.width = 0

            # ── Capture ───────────────────────────────────────────────────────
            if state["capture_flag"]:
                # Use the exact same clean crop here!
                if hand_found and clean_crop.size > 0:
                    rs    = cv2.resize(clean_crop, (200, 200))
                    rgb_c = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
                    inp   = np.expand_dims(rgb_c, axis=0).astype(np.float32)
                    
                    interpreter.set_tensor(inp_d[0]["index"], inp)
                    interpreter.invoke()
                    probs = interpreter.get_tensor(out_d[0]["index"])[0]

                    state["pred_buffer"].append(int(np.argmax(probs)))
                    smoothed_idx = max(
                        set(state["pred_buffer"]),
                        key=list(state["pred_buffer"]).count
                    )
                    confidence = float(probs[smoothed_idx]) * 100
                    letter     = CATEGORIES[smoothed_idx]

                    top3_indices = np.argsort(probs)[::-1][:3]
                    top3 = [(int(i), float(probs[i])*100) for i in top3_indices]

                    if letter in MOTION_SIGNS:
                        page.open(ft.SnackBar(
                            content=ft.Text(f"⚠  Detected '{letter.upper()}' ({confidence:.0f}%) — motion sign, try again.", color=WARNING),
                            bgcolor=SURFACE2,
                        ))
                    elif confidence < CONFIDENCE_THRESHOLD:
                        page.open(ft.SnackBar(
                            content=ft.Text(f"⚠  Low confidence ({confidence:.0f}%) — adjust your hand position.", color=WARNING),
                            bgcolor=SURFACE2,
                        ))
                    elif state["current_tab"] == "quiz":
                        state["attempts"] += 1
                        correct = letter == state["current_prompt"]
                        show_quiz_result(correct, letter, confidence, top3, smoothed_idx)
                    elif state["current_tab"] == "practice":
                        state["attempts"] += 1
                        correct = letter == state["practice_letter"]
                        show_practice_result(correct, letter, confidence, top3, smoothed_idx)
                    else:
                        state["word_letters"].append(letter)
                        word_display.value = " ".join(l.upper() for l in state["word_letters"])
                        word_hint.value = f"{len(state['word_letters'])} letter(s) — last: {letter.upper()} ({confidence:.0f}%)"
                        word_search_btn.visible = True

                else:
                    page.open(ft.SnackBar(
                        content=ft.Text("⚠  No hand in frame — position your hand and try again.", color=WARNING),
                        bgcolor=SURFACE2,
                    ))

                state["capture_flag"]  = False
                state["is_capturing"]  = False
                btn_text.value = "Capture Sign"
                prac_btn_text.value = "Capture Sign"

            # ── Stream frame to UI (rate-limited) ─────────────────────────────
            now = time.time()
            if now - last_ui_update >= 1.0 / UI_FPS:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                camera_img.src_base64 = base64.b64encode(buf).decode("utf-8")
                page.update()
                last_ui_update = now

        cap.release()

    threading.Thread(target=camera_loop, daemon=True).start()


ft.app(target=main)