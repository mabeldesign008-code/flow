import pyperclip
import keyboard
import time
import threading


class TextInjector:
    def inject(self, text: str):
        if not text:
            return

        # Snapshot the user's clipboard
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            old_clipboard = None

        try:
            pyperclip.copy(text)
            time.sleep(0.05)
            keyboard.press_and_release("ctrl+v")
            time.sleep(0.05)
        finally:
            # Restore original clipboard after paste completes
            if old_clipboard is not None:
                threading.Thread(
                    target=self._restore_clipboard,
                    args=(old_clipboard,),
                    daemon=True,
                ).start()

    @staticmethod
    def _restore_clipboard(content: str, delay: float = 0.6):
        time.sleep(delay)
        try:
            pyperclip.copy(content)
        except Exception:
            pass