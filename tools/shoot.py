"""Screenshot every slide at several screen sizes, and run the fake-microphone voice path.

For each size: open the page, show each slide, wait for its animation, save a PNG and check that
the page does not scroll (document scrollHeight <= innerHeight except on the references slide),
that there is no horizontal scroll, and whether a step's text panel overflows. Also checks the
keyboard (→) and the #n address. With fake-microphone WAVs (Chrome's fake capture), records 3 s
on the first step, waits for the voice model and walks the steps with →. Uses the system Chrome.

Usage:
    uv run --no-project --with playwright python tools/shoot.py http://127.0.0.1:8765/ <out dir> [<fake mic wav> ...]
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/usr/bin/google-chrome"
SIZES = {
    "1440x900": dict(viewport={"width": 1440, "height": 900}),
    "1280x720": dict(viewport={"width": 1280, "height": 720}),
    "1920x1080": dict(viewport={"width": 1920, "height": 1080}),
    "390x844": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True),
    "360x740": dict(viewport={"width": 360, "height": 740}, device_scale_factor=2, is_mobile=True, has_touch=True),
}
WAIT = {"hero": 6.5, "divider": 0.6, "input": 2.5, "wave": 4.2, "fft": 7.4, "spec": 4.0, "conv": 5.4, "vector": 4.5, "score": 3.4, "result": 2.8, "refs": 0.6}

METRICS = """() => {
  const d = document.documentElement;
  const p = document.querySelector('#step-panel');
  const scr = [...document.querySelectorAll('.screen')].find(s => !s.hidden);
  const vis = p && !p.closest('.screen').hidden;
  return {
    screen: scr && scr.dataset.screen,
    scene: vis ? document.querySelector('.step:not([hidden])').dataset.scene : null,
    pageScroll: d.scrollHeight - innerHeight,
    hScroll: d.scrollWidth - d.clientWidth,
    screenOverflow: scr ? scr.scrollHeight - scr.clientHeight : 0,
    panelOverflow: vis ? p.scrollHeight - p.clientHeight : 0,
  };
}"""


def open_page(p, url, size, args=(), mic=False):
    browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--autoplay-policy=no-user-gesture-required", *args])
    ctx = browser.new_context(**SIZES[size])
    if mic:
        ctx.grant_permissions(["microphone"])
    page = ctx.new_page()
    logs: list[str] = []
    page.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("window.__birdnetPage && window.__birdnetPage.state.demo && window.__birdnetPage.state.a")
    return browser, page, logs


def slides(page):
    return page.evaluate("window.__birdnetPage.slides.map(s => s.screen === 'steps' ? document.querySelectorAll('.step')[s.step].dataset.scene : s.screen)")


def run_size(p, url, out: Path, size: str) -> dict:
    browser, page, logs = open_page(p, url, size)
    names = slides(page)
    problems = []
    for i, name in enumerate(names):
        page.evaluate(f"window.__birdnetPage.show({i})")
        page.wait_for_timeout(int(WAIT[name] * 1000))
        page.screenshot(path=out / f"{size}-{i + 1:02d}-{name}.png")
        m = page.evaluate(METRICS)
        if m["hScroll"] > 0 or (name != "refs" and (m["pageScroll"] > 0 or m["screenOverflow"] > 0)):
            problems.append({"slide": i + 1, **m})
        elif m["panelOverflow"] > 0:
            problems.append({"slide": i + 1, "note": "text panel scrolls", **m})
    # keyboard and address
    page.evaluate("window.__birdnetPage.show(0)")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    key_ok = page.evaluate("location.hash") == "#2"
    page.goto(url.split("#")[0] + "#5", wait_until="networkidle")
    page.wait_for_timeout(500)
    hash_ok = page.evaluate("document.querySelector('.step:not([hidden])')?.dataset.scene") == names[4]
    browser.close()
    return {"size": size, "slides": len(names), "logs": logs, "problems": problems, "keyboard_ok": key_ok, "hash_ok": hash_ok}


def run_mic(p, url, out: Path, size: str, wav: str, tag: str) -> dict:
    args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={wav}"]
    browser, page, logs = open_page(p, url, size, args, mic=True)
    names = slides(page)
    first = names.index("input")
    page.evaluate(f"window.__birdnetPage.show({first})")
    page.click("#btn-mic")
    page.wait_for_timeout(1700)
    page.screenshot(path=out / f"{tag}-recording.png")
    page.wait_for_function("['ready', 'error', 'quiet'].includes(window.__birdnetPage.state.voice.status)", timeout=60000)
    res = {"voice_status": page.evaluate("window.__birdnetPage.state.voice.status"),
           "p_male": page.evaluate("window.__birdnetPage.state.voice.out && +window.__birdnetPage.state.voice.out.pMale.toFixed(3)")}
    for k, name in enumerate(names[first:first + 8]):
        if k:
            page.keyboard.press("ArrowRight")
        page.wait_for_timeout(int(WAIT[name] * 1000))
        page.screenshot(path=out / f"{tag}-{first + k + 1:02d}-{name}.png")
    # state survives leaving the steps and coming back
    page.evaluate("window.__birdnetPage.show(0)")
    page.wait_for_timeout(300)
    page.evaluate(f"window.__birdnetPage.show({first + 7})")
    res["kept_after_navigation"] = page.evaluate("window.__birdnetPage.state.source === 'mic' && window.__birdnetPage.state.voice.status === 'ready'")
    browser.close()
    return {"tag": tag, "logs": logs, **res}


def main() -> None:
    url, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as p:
        for size in SIZES:
            results.append(run_size(p, url, out, size))
        for k, wav in enumerate(sys.argv[3:]):
            size = ["1440x900", "390x844"][k % 2]
            results.append(run_mic(p, url, out, size, wav, f"mic{k + 1}-{size}"))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
