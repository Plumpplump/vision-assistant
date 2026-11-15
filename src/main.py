# Author: David Li, SISHK

"""Entrypoint. Run with ``flet run`` (desktop) or ``flet run --web``."""

from __future__ import annotations

import flet as ft

from ui import main as ui_main


def main(page: ft.Page) -> None:
    ui_main(page)


if __name__ == "__main__":
    ft.run(main)
