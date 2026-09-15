"""
PlaywrightBrowserEnvironment: a browser-backed ComputerEnvironment.

Requires the optional playwright dependency:
    pip install intelli[computer]
    playwright install chromium
"""

import time

from intelli.function.computer_agent import ComputerEnvironment

# Map Anthropic (xdotool) and OpenAI (uppercase) key names to Playwright names.
_KEY_MAP = {
    "return": "Enter", "enter": "Enter", "kp_enter": "Enter",
    "esc": "Escape", "escape": "Escape",
    "tab": "Tab", "space": " ", "backspace": "Backspace", "delete": "Delete",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "page_down": "PageDown", "pagedown": "PageDown",
    "page_up": "PageUp", "pageup": "PageUp",
    "home": "Home", "end": "End",
    "ctrl": "Control", "control": "Control",
    "alt": "Alt", "shift": "Shift",
    "super": "Meta", "meta": "Meta", "cmd": "Meta",
}


def _normalize_key(combo):
    """Translate 'ctrl+s' / 'CTRL+A' / 'Return' style combos to Playwright syntax."""
    parts = str(combo).replace(" ", "").split("+")
    normalized = []
    for part in parts:
        key = _KEY_MAP.get(part.lower())
        if key is None:
            # Single letters/digits pass through lowercased; longer names title-cased.
            key = part if len(part) == 1 else part.capitalize()
        normalized.append(key)
    return "+".join(normalized)


class PlaywrightBrowserEnvironment(ComputerEnvironment):
    """Drives a Chromium page as the agent's screen."""

    def __init__(self, start_url="about:blank", headless=True, width=1280, height=800):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "PlaywrightBrowserEnvironment requires playwright. "
                "Install with 'pip install intelli[computer]' then 'playwright install chromium'. "
                f"Original error: {e}"
            )
        self.display_width = width
        self.display_height = height
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self.page = self._browser.new_page(viewport={"width": width, "height": height})
        if start_url:
            self.page.goto(start_url)

    def screenshot(self):
        return self.page.screenshot(type="png")

    def click(self, x, y, button="left", modifiers=None):
        if modifiers:
            for m in modifiers:
                self.page.keyboard.down(_normalize_key(m))
        # 'wheel'/'back'/'forward' buttons are not supported by Playwright's mouse.
        btn = button if button in ("left", "right", "middle") else "left"
        self.page.mouse.click(x, y, button=btn)
        if modifiers:
            for m in reversed(modifiers):
                self.page.keyboard.up(_normalize_key(m))

    def double_click(self, x, y):
        self.page.mouse.dblclick(x, y)

    def triple_click(self, x, y):
        self.page.mouse.click(x, y, click_count=3)

    def move(self, x, y):
        self.page.mouse.move(x, y)

    def mouse_down(self, x, y):
        self.page.mouse.move(x, y)
        self.page.mouse.down()

    def mouse_up(self, x, y):
        self.page.mouse.move(x, y)
        self.page.mouse.up()

    def go_back(self):
        self.page.go_back()

    def go_forward(self):
        self.page.go_forward()

    def drag(self, path):
        if not path:
            return
        self.page.mouse.move(*path[0])
        self.page.mouse.down()
        for point in path[1:]:
            self.page.mouse.move(*point)
        self.page.mouse.up()

    def type_text(self, text):
        self.page.keyboard.type(text)

    def key(self, combo):
        self.page.keyboard.press(_normalize_key(combo))

    def hold_key(self, combo, duration):
        key = _normalize_key(combo)
        self.page.keyboard.down(key)
        time.sleep(max(0.0, float(duration)))
        self.page.keyboard.up(key)

    def scroll(self, x, y, scroll_x=0, scroll_y=0, modifiers=None):
        self.page.mouse.move(x, y)
        if modifiers:
            for m in modifiers:
                self.page.keyboard.down(_normalize_key(m))
        self.page.mouse.wheel(scroll_x, scroll_y)
        if modifiers:
            for m in reversed(modifiers):
                self.page.keyboard.up(_normalize_key(m))

    def wait(self, seconds):
        time.sleep(seconds)

    def close(self):
        try:
            self._browser.close()
        finally:
            self._pw.stop()
