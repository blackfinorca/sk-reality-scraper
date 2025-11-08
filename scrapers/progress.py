"""Shared progress helpers for scraper CLIs."""

from __future__ import annotations

from typing import Optional

from tqdm import tqdm


class PercentProgress:
    """Thin wrapper around tqdm that always reports 0–100 progress."""

    def __init__(self, description: str, leave: bool = False) -> None:
        self._bar = tqdm(
            total=100,
            unit="%",
            desc=description,
            leave=leave,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}%",
        )
        self._current = 0.0

    def advance_to(self, percent: float) -> None:
        """Advance the bar to an absolute percentage."""
        bounded = max(0.0, min(100.0, percent))
        if bounded > self._current:
            self._bar.update(bounded - self._current)
            self._current = bounded

    def update_phase(self, completed: int, total: int, phase_start: float, phase_end: float) -> None:
        """Map a sub-task completion ratio into the 0–100 scale."""
        if total <= 0:
            return
        span = max(0.0, phase_end - phase_start)
        ratio = min(1.0, max(0.0, completed / total))
        target = phase_start + span * ratio
        self.advance_to(target)

    def set_postfix(self, text: str) -> None:
        if text:
            self._bar.set_postfix_str(text)

    def close(self) -> None:
        """Complete and close the underlying tqdm bar."""
        self.advance_to(100.0)
        self._bar.close()


def close_progress(progress: Optional[PercentProgress]) -> None:
    if progress is not None:
        progress.close()
