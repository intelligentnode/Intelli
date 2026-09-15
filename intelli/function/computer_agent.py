"""
ComputerAgent: a computer-use loop (screenshot -> model action -> execute -> repeat).

Supports two providers through the existing wrappers:
- anthropic: the native computer-use tool (computer_20251124 / computer_20250124)
  on the Messages API, replaying assistant content verbatim each turn.
- openai: the GA `computer` tool on the Responses API with previous_response_id
  chaining (falls back to the legacy computer_use_preview shape if configured).

The agent acts on a ComputerEnvironment (screen/browser abstraction). Use
PlaywrightBrowserEnvironment from intelli.function.browser_env for web tasks,
or implement the environment interface for a custom desktop.

Usage:
    env = PlaywrightBrowserEnvironment(start_url="https://example.com")
    agent = ComputerAgent(api_key=KEY, provider="anthropic", environment=env)
    result = agent.run("Find the pricing page and report the cheapest plan")
"""

import base64
import time

from intelli.config import config


class ComputerEnvironment:
    """Interface a computer/browser environment must implement for the agent."""

    display_width = 1024
    display_height = 768

    def screenshot(self):
        """Return the current screen as PNG bytes."""
        raise NotImplementedError

    def click(self, x, y, button="left", modifiers=None):
        raise NotImplementedError

    def double_click(self, x, y):
        raise NotImplementedError

    def triple_click(self, x, y):
        # Optional; environments may approximate with a double click.
        self.double_click(x, y)

    def move(self, x, y):
        raise NotImplementedError

    def mouse_down(self, x, y):
        # Fine-grained press; environments that can't split press/release may no-op.
        raise NotImplementedError

    def mouse_up(self, x, y):
        raise NotImplementedError

    def drag(self, path):
        """path: list of (x, y) points from press to release."""
        raise NotImplementedError

    def type_text(self, text):
        raise NotImplementedError

    def key(self, combo):
        """combo: key or key-combo string, e.g. 'Return', 'ctrl+s', 'CTRL+A'."""
        raise NotImplementedError

    def hold_key(self, combo, duration):
        """Hold a key for `duration` seconds. Default approximates with a press."""
        self.key(combo)

    def scroll(self, x, y, scroll_x=0, scroll_y=0, modifiers=None):
        """Scroll at (x, y); positive scroll_y scrolls down. modifiers held during."""
        raise NotImplementedError

    def go_back(self):
        # Browser-style navigation; non-browser environments may no-op.
        pass

    def go_forward(self):
        pass

    def wait(self, seconds):
        time.sleep(seconds)

    def close(self):
        pass


# Anthropic tool versions: newer models use computer_20251124.
_ANTHROPIC_LEGACY_TOKENS = ("sonnet-4-5", "haiku-4-5", "opus-4-1", "sonnet-4-0", "opus-4-0")


def _anthropic_tool_spec(model):
    """Return (tool_type, beta_header) for the given Claude model."""
    m = (model or "").lower()
    if any(token in m for token in _ANTHROPIC_LEGACY_TOKENS):
        return "computer_20250124", "computer-use-2025-01-24"
    return "computer_20251124", "computer-use-2025-11-24"


def _b64_png(png_bytes):
    return base64.standard_b64encode(png_bytes).decode("utf-8")


class ComputerAgent:
    """Runs the screenshot->action loop against Anthropic or OpenAI computer use."""

    def __init__(self, api_key=None, provider="anthropic", model=None, environment=None,
                 max_iterations=25, max_tokens=4096, on_action=None,
                 on_safety_check=None, log=False, options=None):
        """
        Args:
            api_key/provider/model: provider settings; model defaults to the
                configured Anthropic chat default or 'gpt-5.5' for openai.
            environment: a ComputerEnvironment instance (required).
            max_iterations: hard cap on model turns.
            on_action: optional callable(action_dict) -> bool; return False to
                block the action (human-in-the-loop hook). Fail-closed: if it
                raises, the action is treated as blocked.
            on_safety_check: optional callable(call, checks) -> bool for OpenAI
                computer use. It MUST return True to acknowledge the provider's
                pending safety checks and let the loop proceed. Default is to NOT
                acknowledge (the loop stops and surfaces the pending checks), so
                a flagged risk is never auto-approved.
            log: print actions when True.
        """
        if environment is None:
            raise ValueError("ComputerAgent requires an 'environment' instance")
        self.provider = (provider or "anthropic").lower()
        if self.provider not in ("anthropic", "openai"):
            raise ValueError(f"Unsupported computer-use provider: {provider}")
        self.api_key = api_key
        self.env = environment
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.on_action = on_action
        self.on_safety_check = on_safety_check
        self.log = log
        self.options = options or {}
        if self.provider == "anthropic":
            self.model = model or config['url']['anthropic']['models']['chat']
        else:
            self.model = model or "gpt-5.5"
        self._wrapper = None

    # Lazy wrapper creation keeps unit tests free to inject fakes.
    def _get_wrapper(self):
        if self._wrapper is None:
            if self.provider == "anthropic":
                from intelli.wrappers.anthropic_wrapper import AnthropicWrapper
                self._wrapper = AnthropicWrapper(self.api_key)
            else:
                from intelli.wrappers.openai_wrapper import OpenAIWrapper
                self._wrapper = OpenAIWrapper(self.api_key)
        return self._wrapper

    def run(self, task):
        """
        Execute the computer-use task. Returns:
            {"success": bool, "output": str, "iterations": int, "actions": int}
        """
        if self.provider == "anthropic":
            return self._run_anthropic(task)
        return self._run_openai(task)

    def _blocked(self, action):
        """True when the on_action hook rejects this action (fail-closed on error)."""
        if self.on_action is None:
            return False
        try:
            return self.on_action(action) is False
        except Exception:
            # Fail closed: an erroring policy hook must not silently allow actions.
            return True

    def _safety_approved(self, call, checks):
        """True only if the caller explicitly approves the pending safety checks.

        Default (no on_safety_check hook) is to NOT approve, so a provider-flagged
        risk is never auto-acknowledged.
        """
        if self.on_safety_check is None:
            return False
        try:
            return self.on_safety_check(call, checks) is True
        except Exception:
            return False

    # ---------------- Anthropic loop ----------------
    def _run_anthropic(self, task):
        tool_type, beta = _anthropic_tool_spec(self.model)
        tools = [{
            "type": tool_type,
            "name": "computer",
            "display_width_px": self.env.display_width,
            "display_height_px": self.env.display_height,
        }]
        messages = [{"role": "user", "content": task}]
        actions_count = 0

        for iteration in range(1, self.max_iterations + 1):
            params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "tools": tools,
                "messages": messages,
                **self.options,
            }
            response = self._get_wrapper().generate_text(
                params, extra_headers={"anthropic-beta": beta}
            )
            content = response.get("content", [])
            # Replay the assistant content verbatim, as the API requires.
            messages.append({"role": "assistant", "content": content})

            tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            if not tool_uses:
                final = " ".join(b.get("text", "") for b in content
                                 if isinstance(b, dict) and b.get("type") == "text").strip()
                return {"success": True, "output": final, "iterations": iteration, "actions": actions_count}

            # Execute every tool_use and answer all of them in ONE user message.
            results = []
            for block in tool_uses:
                action = block.get("input", {}) or {}
                if self.log:
                    print(f"[computer iter {iteration}] {action.get('action')} {action}")
                if self._blocked(action):
                    results.append({"type": "tool_result", "tool_use_id": block.get("id"),
                                    "content": "Error: action blocked by user policy", "is_error": True})
                    continue
                kind, payload = self._execute_anthropic_action(action)
                if kind == "image":
                    results.append({
                        "type": "tool_result", "tool_use_id": block.get("id"),
                        "content": [{"type": "image",
                                     "source": {"type": "base64", "media_type": "image/png",
                                                "data": _b64_png(payload)}}],
                    })
                else:
                    results.append({"type": "tool_result", "tool_use_id": block.get("id"),
                                    "content": payload})
                actions_count += 1
            messages.append({"role": "user", "content": results})

        return {"success": False, "output": "Stopped: max iterations reached.",
                "iterations": self.max_iterations, "actions": actions_count}

    def _execute_anthropic_action(self, inp):
        """Map an Anthropic computer-use action to environment calls."""
        action = inp.get("action")
        coord = inp.get("coordinate") or [0, 0]
        try:
            if action == "screenshot":
                return "image", self.env.screenshot()
            if action in ("left_click", "right_click", "middle_click"):
                button = action.split("_")[0]
                self.env.click(coord[0], coord[1], button=button,
                               modifiers=[inp["text"]] if inp.get("text") else None)
                return "text", f"{action} at ({coord[0]}, {coord[1]})"
            # Modifier keys the model may hold during a click/scroll.
            modifiers = [inp["text"]] if inp.get("text") else None
            if action == "double_click":
                self.env.double_click(coord[0], coord[1])
                return "text", f"double_click at ({coord[0]}, {coord[1]})"
            if action == "triple_click":
                self.env.triple_click(coord[0], coord[1])
                return "text", f"triple_click at ({coord[0]}, {coord[1]})"
            if action == "mouse_move":
                self.env.move(coord[0], coord[1])
                return "text", f"moved to ({coord[0]}, {coord[1]})"
            if action == "left_mouse_down":
                self.env.mouse_down(coord[0], coord[1])
                return "text", f"mouse down at ({coord[0]}, {coord[1]})"
            if action == "left_mouse_up":
                self.env.mouse_up(coord[0], coord[1])
                return "text", f"mouse up at ({coord[0]}, {coord[1]})"
            if action == "left_click_drag":
                start = inp.get("start_coordinate") or coord
                self.env.drag([tuple(start), tuple(coord)])
                return "text", f"dragged {start} -> {coord}"
            if action == "type":
                self.env.type_text(inp.get("text", ""))
                return "text", "typed text"
            if action == "key":
                self.env.key(inp.get("text", ""))
                return "text", f"pressed {inp.get('text')}"
            if action == "hold_key":
                # Honor the requested hold duration (capped) instead of a single press.
                self.env.hold_key(inp.get("text", ""), min(float(inp.get("duration", 1)), 10))
                return "text", f"held {inp.get('text')}"
            if action == "wait":
                self.env.wait(min(float(inp.get("duration", 1)), 10))
                return "text", "waited"
            if action == "scroll":
                amount = int(inp.get("scroll_amount", 3)) * 100
                direction = inp.get("scroll_direction", "down")
                dx, dy = 0, 0
                if direction == "down":
                    dy = amount
                elif direction == "up":
                    dy = -amount
                elif direction == "right":
                    dx = amount
                elif direction == "left":
                    dx = -amount
                self.env.scroll(coord[0], coord[1], scroll_x=dx, scroll_y=dy, modifiers=modifiers)
                return "text", f"scrolled {direction}"
            return "text", f"Error: unsupported action '{action}'"
        except Exception as e:
            return "text", f"Error executing {action}: {e}"

    # ---------------- OpenAI loop ----------------
    def _run_openai(self, task):
        tools = [{"type": "computer"}]
        payload = {
            "model": self.model,
            "tools": tools,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": task}]}],
            **self.options,
        }
        actions_count = 0

        for iteration in range(1, self.max_iterations + 1):
            response = self._get_wrapper().generate_gpt5_response(payload)
            output = response.get("output", []) or []
            calls = [item for item in output if item.get("type") == "computer_call"]

            if not calls:
                # No more actions: collect the final assistant text.
                texts = []
                for item in output:
                    if item.get("type") == "message":
                        for part in item.get("content", []) or []:
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                texts.append(part.get("text", ""))
                return {"success": True, "output": " ".join(texts).strip(),
                        "iterations": iteration, "actions": actions_count}

            outputs = []
            for call in calls:
                # Provider human-in-the-loop gate: never auto-acknowledge a flagged
                # risk. If there are pending safety checks and the caller does not
                # explicitly approve them, stop and surface them instead of acting.
                pending = call.get("pending_safety_checks") or []
                if pending and not self._safety_approved(call, pending):
                    return {"success": False,
                            "output": "Stopped: pending safety checks require approval.",
                            "iterations": iteration, "actions": actions_count,
                            "pending_safety_checks": pending}

                # GA batches actions[]; the legacy shape has a single action{}.
                actions = call.get("actions") or ([call["action"]] if call.get("action") else [])
                for action in actions:
                    if self.log:
                        print(f"[computer iter {iteration}] {action.get('type')} {action}")
                    if not self._blocked(action):
                        self._execute_openai_action(action)
                        actions_count += 1
                screenshot = self.env.screenshot()
                item = {
                    "type": "computer_call_output",
                    "call_id": call.get("call_id"),
                    "output": {"type": "computer_screenshot",
                               "image_url": f"data:image/png;base64,{_b64_png(screenshot)}",
                               "detail": "original"},
                }
                # Only reached when the checks were explicitly approved above.
                if pending:
                    item["acknowledged_safety_checks"] = pending
                outputs.append(item)

            payload = {
                "model": self.model,
                "tools": tools,
                "previous_response_id": response.get("id"),
                "input": outputs,
                **self.options,
            }

        return {"success": False, "output": "Stopped: max iterations reached.",
                "iterations": self.max_iterations, "actions": actions_count}

    def _execute_openai_action(self, action):
        """Map an OpenAI computer-call action to environment calls."""
        a_type = action.get("type")
        try:
            if a_type == "screenshot":
                return
            if a_type == "click":
                button = action.get("button", "left")
                # 'back'/'forward' are browser navigation, not clicks; 'wheel'
                # is a middle-button press. Route them rather than left-clicking.
                if button == "back":
                    self.env.go_back()
                elif button == "forward":
                    self.env.go_forward()
                else:
                    self.env.click(action.get("x", 0), action.get("y", 0),
                                   button="middle" if button == "wheel" else button,
                                   modifiers=action.get("keys"))
            elif a_type == "double_click":
                self.env.double_click(action.get("x", 0), action.get("y", 0))
            elif a_type == "move":
                self.env.move(action.get("x", 0), action.get("y", 0))
            elif a_type == "drag":
                points = [(p.get("x", p[0] if isinstance(p, (list, tuple)) else 0),
                           p.get("y", p[1] if isinstance(p, (list, tuple)) else 0))
                          if isinstance(p, dict) else tuple(p)
                          for p in action.get("path", [])]
                if points:
                    self.env.drag(points)
            elif a_type == "type":
                self.env.type_text(action.get("text", ""))
            elif a_type == "keypress":
                keys = action.get("keys") or []
                self.env.key("+".join(keys) if isinstance(keys, list) else str(keys))
            elif a_type == "scroll":
                self.env.scroll(action.get("x", 0), action.get("y", 0),
                                scroll_x=action.get("scroll_x", action.get("scrollX", 0)),
                                scroll_y=action.get("scroll_y", action.get("scrollY", 0)))
            elif a_type == "wait":
                self.env.wait(2)
        except Exception:
            # Action errors surface to the model through the next screenshot.
            pass
