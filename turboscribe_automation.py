import argparse
import logging
import os
import re
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = Path(os.getenv("TURBOSCRIBE_PROFILE_DIR", BASE_DIR / "turboscribe_profile"))
DOWNLOAD_DIR = Path(os.getenv("TURBOSCRIBE_DOWNLOAD_DIR", BASE_DIR / "web_jobs" / "turboscribe_downloads"))
DASHBOARD_URL = "https://turboscribe.ai/de/dashboard"
LOGIN_URL = "https://turboscribe.ai/de/login?redirect-url=https%3A%2F%2Fturboscribe.ai%2Fde%2Fdashboard"
DEFAULT_TIMEOUT_MS = int(os.getenv("TURBOSCRIBE_TIMEOUT_MS", "900000"))


def _chrome_path():
    configured = os.getenv("TURBOSCRIBE_CHROME_PATH")
    if configured and os.path.exists(configured):
        return configured

    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    return next((path for path in candidates if path and os.path.exists(path)), None)


def _launch_context(playwright, headless=True):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    launch_options = {
        "headless": headless,
        "accept_downloads": True,
        "viewport": {"width": 1280, "height": 900},
        "downloads_path": str(DOWNLOAD_DIR),
    }
    executable_path = _chrome_path()
    if executable_path:
        launch_options["executable_path"] = executable_path
    return playwright.chromium.launch_persistent_context(str(PROFILE_DIR), **launch_options)


def open_login_browser():
    with sync_playwright() as playwright:
        context = _launch_context(playwright, headless=False)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        logging.info("TurboScribe login browser opened. Log in there, then close the browser window.")
        try:
            page.wait_for_url(re.compile(r".*/dashboard.*"), timeout=DEFAULT_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            logging.info("Login browser closed or timed out before dashboard was detected.")
        context.close()


def ensure_logged_in():
    with sync_playwright() as playwright:
        context = _launch_context(playwright, headless=True)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        url = page.url
        body_text = page.locator("body").inner_text(timeout=5000)
        context.close()

    if "/login" in url or "Anmelden" in body_text:
        raise RuntimeError(
            "TurboScribe login required. Run: venv\\Scripts\\python.exe turboscribe_automation.py login"
        )


def transcribe_to_srt(video_path, output_dir=None):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path(output_dir or DOWNLOAD_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = _launch_context(playwright, headless=True)
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(30000)
        page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
        _raise_if_login_required(page)
        _upload_video(page, video_path)
        _start_transcription(page)
        _open_completed_transcript(page, video_path.stem)
        srt_path = _download_srt(page, output_dir, video_path.stem)
        context.close()
        return str(srt_path)


def _raise_if_login_required(page):
    page.wait_for_timeout(1000)
    body_text = page.locator("body").inner_text(timeout=5000)
    if "/login" in page.url or "Anmelden" in body_text:
        raise RuntimeError(
            "TurboScribe login required. Run: venv\\Scripts\\python.exe turboscribe_automation.py login"
        )


def _upload_video(page, video_path):
    file_input = page.locator('input[type="file"]')
    file_input.wait_for(state="attached", timeout=30000)
    file_input.set_input_files(str(video_path))
    logging.info("Uploaded video to TurboScribe: %s", video_path)
    page.wait_for_timeout(1500)


def _start_transcription(page):
    transcribe_button = page.get_by_text("Transkribieren", exact=False)
    try:
        transcribe_button.click(timeout=30000)
    except PlaywrightTimeoutError:
        page.get_by_role("button", name=re.compile("Transkribieren", re.I)).click(timeout=30000)
    logging.info("Started TurboScribe transcription.")


def _open_completed_transcript(page, source_stem):
    deadline = time.time() + (DEFAULT_TIMEOUT_MS / 1000)
    while time.time() < deadline:
        page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
        _raise_if_login_required(page)
        page.wait_for_timeout(3000)

        body_text = page.locator("body").inner_text(timeout=10000)
        if any(token in body_text.lower() for token in ("transkribiert", "abgeschlossen", "fertig", source_stem.lower())):
            candidates = page.get_by_text(source_stem, exact=False)
            if candidates.count() > 0:
                candidates.first.click()
                page.wait_for_load_state("domcontentloaded", timeout=30000)
                logging.info("Opened TurboScribe transcript page.")
                return

            recent_file = page.locator("a").filter(has_text=re.compile(r".+", re.I))
            if recent_file.count() > 0:
                recent_file.first.click()
                page.wait_for_load_state("domcontentloaded", timeout=30000)
                logging.info("Opened first recent TurboScribe transcript page.")
                return

        logging.info("Waiting for TurboScribe transcription to finish...")
        page.wait_for_timeout(10000)

    raise TimeoutError("TurboScribe transcription did not finish in time.")


def _download_srt(page, output_dir, source_stem):
    _click_export_or_download(page)
    _choose_srt_format(page)

    with page.expect_download(timeout=60000) as download_info:
        _click_final_download(page)
    download = download_info.value
    suggested_name = download.suggested_filename or f"{source_stem}.srt"
    if not suggested_name.lower().endswith(".srt"):
        suggested_name = f"{Path(suggested_name).stem}.srt"

    target_path = output_dir / _safe_filename(suggested_name)
    download.save_as(str(target_path))
    logging.info("Downloaded TurboScribe SRT: %s", target_path)
    return target_path


def _click_export_or_download(page):
    labels = ["Exportieren", "Herunterladen", "Download", "Export", "Untertitel"]
    for label in labels:
        locator = page.get_by_text(label, exact=False)
        try:
            if locator.count() > 0:
                locator.first.click(timeout=5000)
                page.wait_for_timeout(800)
                return
        except PlaywrightTimeoutError:
            continue
    raise RuntimeError("Could not find TurboScribe export/download button.")


def _choose_srt_format(page):
    labels = ["SRT", ".srt", "Untertitel"]
    for label in labels:
        locator = page.get_by_text(label, exact=False)
        try:
            if locator.count() > 0:
                locator.first.click(timeout=5000)
                page.wait_for_timeout(500)
                return
        except PlaywrightTimeoutError:
            continue


def _click_final_download(page):
    labels = ["Herunterladen", "Download", "Exportieren", "Download ZIP"]
    for label in labels:
        locator = page.get_by_text(label, exact=False)
        try:
            if locator.count() > 0:
                locator.first.click(timeout=5000)
                return
        except PlaywrightTimeoutError:
            continue
    raise RuntimeError("Could not find final TurboScribe download button.")


def _safe_filename(filename):
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename).strip("._")
    return cleaned or "turboscribe.srt"


def main():
    parser = argparse.ArgumentParser(description="TurboScribe browser automation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login", help="Open the persistent TurboScribe login browser")
    subparsers.add_parser("check-login", help="Check whether the persistent profile is logged in")

    transcribe_parser = subparsers.add_parser("transcribe", help="Upload a video and download SRT")
    transcribe_parser.add_argument("video_path")
    transcribe_parser.add_argument("--output-dir", default=str(DOWNLOAD_DIR))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "login":
        open_login_browser()
    elif args.command == "check-login":
        ensure_logged_in()
        print("TurboScribe profile is logged in.")
    elif args.command == "transcribe":
        print(transcribe_to_srt(args.video_path, args.output_dir))


if __name__ == "__main__":
    main()
