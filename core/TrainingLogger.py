from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainingLogger:
    FIELDNAMES = [
        "run_id",
        "timestamp",
        "agent_name",
        "map_name",
        "mode",
        "generation",
        "completed_generations",
        "generations_total",
        "best_score",
        "avg_score",
        "median_score",
        "crashed_count",
        "cars_total",
        "cars_active",
        "best_score_all",
        "stagnation_epochs",
        "time_s",
        "max_time_s",
        "save_as",
        "load_from",
    ]

    def __init__(
        self,
        log_dir: str | Path,
        plot_dir: str | Path,
        *,
        plot_every: int = 5,
        agent_name: str = "",
        map_name: str = "",
        save_as: str = "",
        load_from: str = "",
        mode: str = "start",
    ):
        self.log_dir = Path(log_dir)
        self.plot_dir = Path(plot_dir)
        self.plot_every = max(0, int(plot_every or 0))
        self.agent_name = str(agent_name or "<agent>")
        self.map_name = str(map_name or "<map>")
        self.save_as = str(save_as or "")
        self.load_from = str(load_from or "")
        self.mode = str(mode or "start")

        now = datetime.now()
        run_parts = [
            now.strftime("%Y%m%d_%H%M%S_%f"),
            self._safe_name(self.agent_name, "agent"),
            self._safe_name(Path(self.map_name).stem, "map"),
            self._safe_name(Path(self.save_as).stem, "save"),
        ]
        self.run_id = "_".join(part for part in run_parts if part)
        self.csv_path = self.log_dir / f"{self.run_id}.csv"
        self.plot_path = self.plot_dir / f"{self.run_id}.png"
        self.rows: list[dict[str, Any]] = []
        self.csv_enabled = True
        self._matplotlib_warning_printed = False
        self._plot_error_printed = False

        self._prepare_csv()
        print(f"Training CSV log: {self.csv_path}")
        if self.plot_every > 0:
            print(f"Training PNG graf: {self.plot_path} (kazdych {self.plot_every} generaci)")
        else:
            print("Training PNG graf je vypnuty")

    @staticmethod
    def _safe_name(text: str, default: str) -> str:
        text = str(text or "").strip() or default
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
        text = text.strip("._-")
        return (text or default)[:80]

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _prepare_csv(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
        except Exception as exc:
            self.csv_enabled = False
            print(f"VAROVANI: CSV log treninku se nepodarilo pripravit: {exc}")

    def log_generation(self, stats: dict[str, Any]) -> None:
        row = self._make_row(stats)
        self.rows.append(row)
        self._append_csv(row)

        if self.plot_every > 0 and len(self.rows) % self.plot_every == 0:
            self.save_plot()

    def save_plot(self) -> bool:
        if not self.rows or self.plot_every <= 0:
            return False
        if self._matplotlib_warning_printed:
            return False

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            if not self._matplotlib_warning_printed:
                print(f"INFO: matplotlib neni dostupny, PNG graf treninku se neulozi: {exc}")
                self._matplotlib_warning_printed = True
            return False

        try:
            self.plot_dir.mkdir(parents=True, exist_ok=True)

            generations = [self._as_int(row["completed_generations"]) for row in self.rows]
            best_scores = [self._as_float(row["best_score"]) for row in self.rows]
            avg_scores = [self._as_float(row["avg_score"]) for row in self.rows]
            median_scores = [self._as_float(row["median_score"]) for row in self.rows]
            crashes = [self._as_int(row["crashed_count"]) for row in self.rows]

            fig, score_ax = plt.subplots(figsize=(9, 5))
            score_ax.plot(generations, best_scores, label="Best score", color="#1f77b4", linewidth=2)
            score_ax.plot(generations, avg_scores, label="Average score", color="#2ca02c", linewidth=1.6)
            score_ax.plot(generations, median_scores, label="Median score", color="#ff7f0e", linewidth=1.6)
            score_ax.set_xlabel("Generation")
            score_ax.set_ylabel("Score")
            score_ax.grid(True, alpha=0.25)

            crash_ax = score_ax.twinx()
            crash_ax.bar(generations, crashes, label="Crashes", color="#d62728", alpha=0.18)
            crash_ax.set_ylabel("Crashes")

            title = f"{self.agent_name} on {self.map_name}"
            score_ax.set_title(title)
            lines, labels = score_ax.get_legend_handles_labels()
            bars, bar_labels = crash_ax.get_legend_handles_labels()
            score_ax.legend(lines + bars, labels + bar_labels, loc="best")

            fig.tight_layout()
            fig.savefig(self.plot_path, dpi=130)
            plt.close(fig)
            print(f"Training PNG graf ulozen: {self.plot_path}")
            return True
        except Exception as exc:
            if not self._plot_error_printed:
                print(f"VAROVANI: PNG graf treninku se nepodarilo ulozit: {exc}")
                self._plot_error_printed = True
            return False

    def _append_csv(self, row: dict[str, Any]) -> None:
        if not self.csv_enabled:
            return

        try:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)
        except Exception as exc:
            self.csv_enabled = False
            print(f"VAROVANI: CSV log treninku se nepodarilo zapsat: {exc}")

    def _make_row(self, stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent_name": self.agent_name,
            "map_name": self.map_name,
            "mode": self.mode,
            "generation": self._as_int(stats.get("generation")),
            "completed_generations": self._as_int(stats.get("completed_generations")),
            "generations_total": self._as_int(stats.get("generations_total")),
            "best_score": round(self._as_float(stats.get("last_generation_best_score")), 6),
            "avg_score": round(self._as_float(stats.get("last_generation_avg_score")), 6),
            "median_score": round(self._as_float(stats.get("last_generation_median_score")), 6),
            "crashed_count": self._as_int(stats.get("last_generation_crashed_count")),
            "cars_total": self._as_int(stats.get("cars_total")),
            "cars_active": self._as_int(stats.get("cars_active")),
            "best_score_all": round(self._as_float(stats.get("best_score_all")), 6),
            "stagnation_epochs": self._as_int(stats.get("stagnation_epochs")),
            "time_s": round(self._as_float(stats.get("time")), 6),
            "max_time_s": round(self._as_float(stats.get("max_time")), 6),
            "save_as": self.save_as,
            "load_from": self.load_from,
        }
