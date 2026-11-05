"""Record the README demo GIF from the running Streamlit app.

Drives the UI in headless Chrome, screenshots each step and assembles the
frames into `docs/demo.gif`: the question is typed, Search is pressed, the
cited answer appears, and the page scrolls to the first source, opened.

Two things about the result are worth knowing:

- The wait for the agent is cut down to a few spinner frames. The latency
  shown in the app's own metrics column is the real one for that query.
- One untimed query runs first, so the recording shows a warm app (models
  loaded), which is the state the eval latencies were measured in.

Needs the app running and Playwright, which is not a project dependency. It
drives the installed Google Chrome, so no browser download is required. The
committed GIF was recorded with the app in the light theme and with the
developer toolbar hidden, as a visitor to the deployed app sees it:

    streamlit run src/medrag/ui/app.py --theme.base light --client.toolbarMode viewer
    pip install playwright
    python scripts/record_demo.py

Usage:
    python scripts/record_demo.py
    python scripts/record_demo.py --url http://localhost:8501 --question "..."
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from playwright.sync_api import Page

QUESTION = "What did the SELECT trial show about semaglutide in non-diabetic obesity?"
WARMUP_QUESTION = "What was the primary cardiovascular outcome in the LEADER trial?"

VIEWPORT = {"width": 1120, "height": 840}
ANSWER_TIMEOUT_MS = 180_000

# Frame durations in milliseconds.
TYPING_MS = 70
PAUSE_MS = 700
SPINNER_MS = 220
SPINNER_FRAMES = 7
ANSWER_HOLD_MS = 4500
SCROLL_MS = 60
SCROLL_STEPS = 8
SOURCE_HOLD_MS = 4000


@dataclass
class Frame:
    image: Image.Image
    duration_ms: int


def _shot(page: Page, duration_ms: int) -> Frame:
    png = page.screenshot(type="png")
    return Frame(Image.open(io.BytesIO(png)).convert("RGB"), duration_ms)


def _ask(page: Page, url: str, question: str) -> None:
    """Run one query to completion without recording it."""
    page.goto(url)
    box = page.get_by_label("Clinical question")
    box.fill(question)
    box.press("Enter")
    page.get_by_role("button", name="Search", exact=True).click()
    page.get_by_role("heading", name="Answer").wait_for(timeout=ANSWER_TIMEOUT_MS)


def record(page: Page, url: str, question: str) -> list[Frame]:
    page.goto(url)
    box = page.get_by_label("Clinical question")
    box.wait_for()
    page.wait_for_timeout(1500)
    frames = [_shot(page, PAUSE_MS)]

    box.click()
    for i, char in enumerate(question, start=1):
        page.keyboard.type(char)
        if i % 2 == 0 or i == len(question):
            frames.append(_shot(page, TYPING_MS))
    frames[-1].duration_ms = PAUSE_MS

    box.press("Enter")
    search = page.get_by_role("button", name="Search", exact=True)
    search.wait_for()
    page.wait_for_timeout(400)
    search.click()

    answer = page.get_by_role("heading", name="Answer")
    spinner: list[Frame] = []
    while not answer.is_visible():
        spinner.append(_shot(page, SPINNER_MS))
        page.wait_for_timeout(SPINNER_MS)
        if len(spinner) * SPINNER_MS > ANSWER_TIMEOUT_MS:
            raise TimeoutError("the app did not answer in time")
    frames.extend(spinner[:SPINNER_FRAMES])

    # Let the metrics column and the source list finish rendering.
    page.get_by_text("Chunks considered").wait_for()
    page.wait_for_timeout(1000)
    frames.append(_shot(page, ANSWER_HOLD_MS))

    # Open the first source, then scroll until "Sources" sits near the top.
    page.locator("details summary").first.click()
    page.wait_for_timeout(600)
    heading = page.get_by_role("heading", name="Sources").bounding_box()
    distance = max(0.0, heading["y"] - 90) if heading else 0.0
    page.mouse.move(VIEWPORT["width"] * 0.6, VIEWPORT["height"] * 0.5)
    for _ in range(SCROLL_STEPS):
        page.mouse.wheel(0, distance / SCROLL_STEPS)
        page.wait_for_timeout(120)
        frames.append(_shot(page, SCROLL_MS))
    page.wait_for_timeout(500)
    frames.append(_shot(page, SOURCE_HOLD_MS))
    return frames


def write_gif(frames: list[Frame], out: Path, colors: int) -> None:
    """Quantise every frame to one shared palette, so flat areas do not flicker."""
    # The last frame shows the most UI (answer, metrics, an open source).
    palette = frames[-1].image.quantize(colors=colors, dither=Image.Dither.NONE)
    quantised = [f.image.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        out,
        save_all=True,
        append_images=quantised[1:],
        duration=[f.duration_ms for f in frames],
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    parser.add_argument("--colors", type=int, default=64, help="palette size (fewer = smaller)")
    parser.add_argument("--no-warmup", action="store_true", help="record the cold first query")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport=VIEWPORT, color_scheme="light")
        page = context.new_page()
        if not args.no_warmup:
            _ask(page, args.url, WARMUP_QUESTION)
        frames = record(page, args.url, args.question)
        browser.close()

    write_gif(frames, args.out, args.colors)
    seconds = sum(f.duration_ms for f in frames) / 1000
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}: {len(frames)} frames, {seconds:.1f} s, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
