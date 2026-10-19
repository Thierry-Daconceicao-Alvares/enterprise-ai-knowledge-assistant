"""Capture the README screenshots and the demo GIF from the running application.

Usage (needs GEMINI_API_KEY, and Playwright with Chromium):
    pip install playwright==1.64.0 && playwright install chromium
    python scripts/capture_screenshots.py --output docs/images

The script starts the Streamlit app on a fresh index in a temporary folder,
loads the sample documents, asks the example questions, and saves what a
reviewer would see. The answers are real model outputs: nothing is staged.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
STATUS_TEXTS = ["Answered from the documents", "Not enough evidence", "Clarification needed"]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url: str, timeout_s: float = 90) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"{url} did not answer within {timeout_s} s")


def wait_for_answer(page: Page, count: int, timeout_ms: int = 240_000) -> bool:
    """Wait until `count` answers are displayed. Returns False when the app
    shows an error instead (for example a free-tier rate limit)."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if sum(page.get_by_text(text).count() for text in STATUS_TEXTS) >= count:
            page.wait_for_timeout(1500)
            return True
        if page.get_by_text("temporarily unavailable").count():
            return False
        page.wait_for_timeout(500)
    raise TimeoutError("no answer displayed")


def ask_until_answered(page: Page, question_start: str, attempts: int = 3) -> None:
    """Ask an example question; after a rate-limit error, wait and ask again."""
    for attempt in range(1, attempts + 1):
        ask_example(page, question_start)
        if wait_for_answer(page, 1):
            return
        print(f"Model unavailable for {question_start!r} (attempt {attempt}); waiting 60 s")
        clear(page)
        page.wait_for_timeout(60_000)
    raise RuntimeError(f"No answer for {question_start!r} after {attempts} attempts")


def ask_example(page: Page, question_start: str) -> None:
    page.get_by_role("button", name=question_start).first.click()


def clear(page: Page) -> None:
    page.get_by_role("button", name="Clear the conversation").click()
    page.wait_for_timeout(1500)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set.")

    workdir = Path(tempfile.mkdtemp())
    data_dir = workdir / "data"
    shutil.copytree(ROOT / "data" / "sample_documents", data_dir / "sample_documents")
    port = free_port()
    env = {**os.environ, "DATA_DIR": str(data_dir), "PUBLIC_DEMO": "false", "LOG_LEVEL": "WARNING"}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app" / "streamlit_app.py"),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        env=env,
        cwd=ROOT,
    )
    frames: list[Path] = []
    try:
        wait_for(f"http://127.0.0.1:{port}/_stcore/health")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=os.environ.get("CHROMIUM_PATH") or None
            )
            page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.get_by_role("button", name="Load the 32 sample documents").click()
            page.get_by_text("Try an example").wait_for(timeout=180_000)
            page.wait_for_timeout(1500)

            def shot(name: str, full_page: bool = False) -> Path:
                path = args.output / f"{name}.png"
                page.screenshot(path=str(path), full_page=full_page)
                return path

            frames.append(shot("01-home"))

            ask_until_answered(page, "What is the procedure to approve a purchase request")
            frames.append(shot("02-answer"))
            page.locator('[data-testid="stExpander"]').filter(has_text="relevance").first.click()
            page.wait_for_timeout(1000)
            shot("03-answer-with-sources", full_page=True)
            toggle = page.get_by_text("Show page").first
            if toggle.count():
                toggle.click()
                page.wait_for_timeout(2500)
                toggle.scroll_into_view_if_needed()
                frames.append(shot("04-source-page"))

            for name, question in [
                ("05-not-in-documents", "What is the daily meal allowance"),
                ("06-contradiction", "What is the maximum I can spend on dinner"),
                ("07-hidden-injection", "Do I need quotes and a purchase order"),
                ("08-outdated-version", "How often do I have to change my password"),
            ]:
                clear(page)
                ask_until_answered(page, question)
                frames.append(shot(name))
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
        shutil.rmtree(workdir, ignore_errors=True)

    images = [Image.open(frame).convert("RGB") for frame in frames]
    width = 1000
    resized = [image.resize((width, int(image.height * width / image.width))) for image in images]
    resized[0].save(
        args.output / "demo.gif",
        save_all=True,
        append_images=resized[1:],
        duration=3200,
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(frames)} screenshots and demo.gif to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
