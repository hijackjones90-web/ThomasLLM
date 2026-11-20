"""Output emitters: OSD overlay, global hotkey, named pipe / socket."""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

try:
    import tkinter as tk
    from tkinter import font as tkfont
    _HAS_TKINTER = True
except Exception:  # pragma: no cover - tkinter optional
    tk = None
    tkfont = None
    _HAS_TKINTER = False

try:
    from pynput import keyboard
    _HAS_PYNPUT = True
except Exception:  # pragma: no cover - pynput optional
    keyboard = None
    _HAS_PYNPUT = False

try:
    import win32pipe
    import win32file
    import pywintypes
    _HAS_PYWIN32 = True
except Exception:  # pragma: no cover - pywin32 optional
    win32pipe = None
    win32file = None
    pywintypes = None
    _HAS_PYWIN32 = False

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# OSD Overlay
# --------------------------------------------------------------------------- #
class OverlayEmitter:
    """Click-through, top-most overlay window showing the latest response."""

    def __init__(
        self,
        text: str = "",
        position: str = "bottom-right",
        width: int = 400,
        height: int = 100,
        font_size: int = 12,
        bg_color: str = "#000000",
        fg_color: str = "#ffffff",
        alpha: float = 0.85,
    ):
        if not _HAS_TKINTER:
            raise RuntimeError("tkinter not available; install python3-tk (Linux) or use Windows default Python")
        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._text = text
        self._position = position
        self._width = width
        self._height = height
        self._font_size = font_size
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._alpha = alpha

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
            self._root = None

    def update(self, text: str) -> None:
        self._text = text
        if self._root and self._label:
            try:
                self._root.after(0, lambda: self._label.config(text=self._text))
            except Exception:
                pass

    def _run(self) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", self._alpha)
        # Windows click-through
        try:
            self._root.attributes("-transparentcolor", self._bg_color)
        except Exception:
            pass

        self._root.geometry(f"{self._width}x{self._height}")
        self._position_window()

        self._label = tk.Label(
            self._root,
            text=self._text,
            fg=self._fg_color,
            bg=self._bg_color,
            font=tkfont.nametofont("TkDefaultFont").copy().configure(size=self._font_size),
            wraplength=self._width - 20,
            justify="left",
        )
        self._label.pack(fill="both", expand=True, padx=10, pady=10)

        self._root.mainloop()

    def _position_window(self) -> None:
        if not self._root:
            return
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        if self._position == "bottom-right":
            x = sw - self._width - 20
            y = sh - self._height - 60
        elif self._position == "top-right":
            x = sw - self._width - 20
            y = 20
        elif self._position == "bottom-left":
            x = 20
            y = sh - self._height - 60
        else:  # center
            x = (sw - self._width) // 2
            y = (sh - self._height) // 2
        self._root.geometry(f"+{x}+{y}")


# --------------------------------------------------------------------------- #
# Global Hotkey
# --------------------------------------------------------------------------- #
class HotkeyEmitter:
    """Global hotkey listener (pynput). Calls a callback on activation."""

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        if not _HAS_PYNPUT:
            raise RuntimeError("pynput not installed; `pip install pynput`")
        self._hotkey = hotkey
        self._callback = callback
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._listener:
            return
        self._listener = keyboard.GlobalHotKeys({self._hotkey: self._on_activate})
        self._thread = threading.Thread(target=self._listener.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _on_activate(self) -> None:
        log.debug("Hotkey %s activated", self._hotkey)
        try:
            self._callback()
        except Exception as e:
            log.exception("Hotkey callback error: %s", e)


# --------------------------------------------------------------------------- #
# Named Pipe / Socket Emitter
# --------------------------------------------------------------------------- #
class PipeEmitter:
    """Writes JSON Lines to a named pipe (Windows) or TCP socket."""

    def __init__(
        self,
        pipe_path: str = r"\\.\pipe\qwen35",
        socket_port: Optional[int] = 8765,
        use_pipe: bool = True,
    ):
        self._pipe_path = pipe_path
        self._socket_port = socket_port
        self._use_pipe = use_pipe and _HAS_PYWIN32
        self._pipe_handle = None
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False

    def start(self) -> None:
        if self._connected:
            return
        if self._use_pipe:
            self._connect_pipe()
        else:
            self._connect_socket()
        self._connected = True

    def stop(self) -> None:
        self._connected = False
        if self._pipe_handle:
            try:
                win32file.CloseHandle(self._pipe_handle)
            except Exception:
                pass
            self._pipe_handle = None
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _connect_pipe(self) -> None:
        """Open the named pipe for writing."""
        while self._connected:
            try:
                self._pipe_handle = win32file.CreateFile(
                    self._pipe_path,
                    win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )
                log.info("Connected to named pipe %s", self._pipe_path)
                return
            except pywintypes.error as e:
                if e.winerror == 231:  # ERROR_PIPE_BUSY
                    time.sleep(0.1)
                    continue
                if e.winerror == 2:  # FILE_NOT_FOUND
                    log.debug("Pipe %s not yet created; retrying...", self._pipe_path)
                    time.sleep(0.5)
                    continue
                raise
        raise RuntimeError("Pipe connection cancelled")

    def _connect_socket(self) -> None:
        """Connect to localhost TCP port."""
        while self._connected:
            try:
                self._socket = socket.create_connection(("127.0.0.1", self._socket_port), timeout=2)
                log.info("Connected to socket 127.0.0.1:%d", self._socket_port)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Socket connection cancelled")

    def emit(self, data: Dict[str, Any]) -> None:
        """Serialize dict to JSON Line and write to pipe/socket."""
        if not self._connected:
            return
        line = json.dumps(data, ensure_ascii=False) + "\n"
        payload = line.encode("utf-8")
        with self._lock:
            try:
                if self._use_pipe and self._pipe_handle:
                    win32file.WriteFile(self._pipe_handle, payload)
                elif self._socket:
                    self._socket.sendall(payload)
            except Exception as e:
                log.warning("Emit failed (pipe/socket closed?): %s", e)
                self._connected = False  # trigger reconnect on next emit


# --------------------------------------------------------------------------- #
# Socket Server (for consumers to connect to)
# --------------------------------------------------------------------------- #
class SocketServer:
    """Simple TCP server that accepts connections and echoes JSON lines (for testing)."""

    def __init__(self, port: int = 8765):
        self._port = port
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self._port))
        self._server.listen(5)
        log.info("Socket server listening on port %d", self._port)
        self._server.settimeout(1.0)
        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
                log.debug("Client connected: %s", addr)
                conn.close()  # simple echo server; real impl would handle streams
            except socket.timeout:
                continue
            except Exception:
                if not self._stop.is_set():
                    log.exception("Socket server error")
        self._server.close()


# --------------------------------------------------------------------------- #
# Composite Emitter Manager
# --------------------------------------------------------------------------- #
class EmitterManager:
    """Convenience wrapper that starts/stops all configured emitters."""

    def __init__(self, app_cfg: Any):
        self.overlay: Optional[OverlayEmitter] = None
        self.hotkey: Optional[HotkeyEmitter] = None
        self.pipe: Optional[PipeEmitter] = None
        self.socket_server: Optional[SocketServer] = None
        self._app_cfg = app_cfg
        self._hotkey_callback: Optional[Callable[[], None]] = None

    def set_hotkey_callback(self, callback: Callable[[], None]) -> None:
        self._hotkey_callback = callback

    def start(self) -> None:
        oc = self._app_cfg.outputs
        hc = self._app_cfg.hotkey
        cc = self._app_cfg.capture

        if oc.overlay:
            self.overlay = OverlayEmitter(
                position="bottom-right",
                width=400,
                height=100,
                font_size=12,
            )
            self.overlay.start()

        if hc.enabled and self._hotkey_callback:
            self.hotkey = HotkeyEmitter(hc.hotkey, self._hotkey_callback)
            self.hotkey.start()

        if oc.emit_responses:
            self.pipe = PipeEmitter(
                pipe_path=oc.pipe_path,
                socket_port=oc.socket_port,
                use_pipe=True,
            )
            self.pipe.start()

        if oc.socket_port:
            self.socket_server = SocketServer(oc.socket_port)
            self.socket_server.start()

    def stop(self) -> None:
        for emitter in (self.overlay, self.hotkey, self.pipe, self.socket_server):
            if emitter:
                emitter.stop()

    def emit_response(self, response: str, source: str, frame_id: int, latency_ms: float) -> None:
        """Emit a structured JSON response to all active sinks."""
        data = {
            "ts": time.time(),
            "source": source,  # "daemon" or "hotkey"
            "frame_id": frame_id,
            "latency_ms": latency_ms,
            "response": response,
        }
        if self.overlay:
            self.overlay.update(response)
        if self.pipe:
            self.pipe.emit(data)