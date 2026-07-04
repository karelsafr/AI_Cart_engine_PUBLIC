from __future__ import annotations

import os

import pygame
from my_sprites.AI_car import AI_car
from constants import (
    WHITE, SPEED, MAX_SPEED, TILESIZE, TURN_SPEED, BREAK_SPEED, FRICTION_SPEED,
    PATH_SAVES, PATH_TRAINING_LOGS, PATH_TRAINING_PLOTS,
    TRAINING_PLOT_EVERY
)
from core.TrainingLogger import TrainingLogger
import copy
from pathlib import Path
from typing import Any, Callable
import numpy as np
from core.protocols import AgentProtocol

SPAWN_Y_JITTER_PX = 12

class Car_manager():
    def __init__(self, pocet_aut: int, pocet_generaci: int, max_time: float, cars_to_next: int,save_as: str, load_from: str):
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from

        self.epoch = 0
        self.cur_epoch = 0
        self.total_time = 0
        self.running = False # tykas se zda bezi nejaká epocha
        self.pustene = False # týká se zda bezí celkvoe trening
        self.paused = False

        self.sprite_list: list[AI_car] = list() # registrujeme do vsech - abych je pak byl schopen prebrat
        self.sprite_running = pygame.sprite.Group() # praucjeme jen s running ve kole
        self.best_cars_list: list[AI_car] = list() # bezst brain z minulé epochy
        self.training_logger: TrainingLogger | None = None
        self.defaultbrain: Callable[[], AgentProtocol]
        self.brain_list: list[AgentProtocol] = []
        self.run_agent_name = ""
        self.run_map_name = ""
        self._normalize_settings()
        self.reset_monitoring()

    def _clear_cars(self) -> None:
        for c in list(self.sprite_list):
            c.kill()
        self.sprite_list = list()
        self.sprite_running = pygame.sprite.Group()

    def setup(self, pocet_aut: int, pocet_generaci: int, max_time: float, cars_to_next: int, save_as: str, load_from: str) -> None:
        self.pocet_aut = pocet_aut
        self.pocet_generaci = pocet_generaci
        self.max_time = max_time
        self.cars_to_next = cars_to_next
        self.save_as = save_as
        self.load_from = load_from
        self._normalize_settings()

    def _normalize_settings(self) -> None:
        self.pocet_aut = max(1, int(self.pocet_aut or 1))
        self.pocet_generaci = max(0, int(self.pocet_generaci or 0))
        self.max_time = max(0.1, float(self.max_time or 0.1))

        requested_cars_to_next = int(self.cars_to_next or 1)
        normalized_cars_to_next = max(1, min(requested_cars_to_next, self.pocet_aut))
        if requested_cars_to_next != normalized_cars_to_next:
            print(
                "VAROVANI: cars_to_next musi byt mezi 1 a pocet_aut. "
                f"Pouzivam {normalized_cars_to_next}."
            )
        self.cars_to_next = normalized_cars_to_next

    def set_run_metadata(self, agent_name: str = "", map_name: str = "") -> None:
        self.run_agent_name = str(agent_name or "").strip()
        self.run_map_name = str(map_name or "").strip()

    def _start_training_logger(self, mode: str) -> None:
        self.training_logger = TrainingLogger(
            PATH_TRAINING_LOGS,
            PATH_TRAINING_PLOTS,
            plot_every=TRAINING_PLOT_EVERY,
            agent_name=self.run_agent_name,
            map_name=self.run_map_name,
            save_as=self.save_as,
            load_from=self.load_from,
            mode=mode,
        )

    def setup_next_epoch(self) -> None:
        self.score_cars()# oskoruji auta
        self.record_generation_stats()

        parents_to_keep = min(self.cars_to_next, len(self.best_cars_list))
        if parents_to_keep < 1:
            print("ERROR: nelze vytvorit dalsi generaci, nejsou dostupna zadna auta")
            self.running = False
            self.pustene = False
            return

        if parents_to_keep != self.cars_to_next:
            print(
                "VAROVANI: cars_to_next je vetsi nez aktualni pocet aut v generaci. "
                f"Pro tento prechod pouzivam {parents_to_keep}, dalsi generace ponecha nastavene {self.cars_to_next}."
            )

        self.best_cars_list = [c for c in self.best_cars_list[0:parents_to_keep]]# ulozim si strnaou nejlepsích n

        self.sprite_list = list()# vyresetuji cekový list
        for c in self.sprite_running:# vymazu vse
            c.kill()
        self.reset_brains()
        self.total_time = 0
        self.cur_epoch += 1
        self.paused = False

        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        for i in range(parents_to_keep):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20,
                       copy.deepcopy(self.best_cars_list[i].brain),  180+np.random.randint(-45,+45))
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # doplneneí zmutovanou generací prvních n aut:
        for j in range(self.pocet_aut-parents_to_keep):
            i = j % parents_to_keep
            c = AI_car(TILESIZE * 4 + int(TILESIZE / 2), TILESIZE * 8 + int(TILESIZE / 2), 10, 20,
                       copy.deepcopy(self.best_cars_list[i].brain))

            c.brain.mutate()
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        self.running = True

    def score_cars(self) -> None:
        self.best_cars_list = list()
        self.best_cars_list = sorted(self.sprite_list, key=lambda obj: obj.brain.score, reverse=True)

    def add_defaultbrain(self, brain: Callable[[], AgentProtocol]) -> None:
        self.defaultbrain = brain
        self.reset_brains()

    # start - od začátku vše, zresetuje co jde a nastaví auta znova
    def start(self) -> None:
        # zapnu start a vymazu vsechny informace pokud nekde jsou:
        self._clear_cars()
        self.training_logger = None

        self.cur_epoch = 0
        self.total_time = 0
        self.paused = False
        self.reset_monitoring()

        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        # a vytvořím auta:
        for i in range(self.pocet_aut):
            y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
            c = AI_car(base_x, y, 10, 20, self.brain_list[i], 180+np.random.randint(-45,+45))
            self.sprite_list.append(c)
            self.sprite_running.add(c)

        # a zapnu tréning
        self._start_training_logger("start")
        self.running = True
        self.pustene = True

    def get_sprite_group(self) -> pygame.sprite.Group:
        return self.sprite_running

    def reset_brains(self) -> None:
        self.brain_list = [self.defaultbrain() for _ in range(self.pocet_aut)]

    def draw(self, screen: pygame.Surface) -> None:
        for sprite in self.sprite_list:
            screen.blit(sprite.image, sprite.rect)

    def stop(self) -> None:
        self.running = False
        self.paused = False

    def toggle_pause(self) -> bool:
        if not self.pustene or not self.sprite_list:
            print("PAUSE/RESUME: neni spusteny zadny trening")
            return False

        if self.paused:
            self.paused = False
            self.running = True
            print("Trening pokracuje")
        else:
            self.paused = True
            print("Trening pozastaven")
        return True

    def reset_run(self) -> bool:
        self.running = False
        self.pustene = False
        self.paused = False
        self.cur_epoch = 0
        self.total_time = 0
        self._clear_cars()
        self.best_cars_list = list()
        self.training_logger = None
        self.reset_monitoring()
        print("Trening resetovan")
        return True

    def next_generation(self) -> bool:
        if not self.pustene or not self.sprite_list:
            print("NEXT GEN: neni spusteny zadny trening")
            return False
        print(f"NEXT GEN: rucne ukoncuji epochu {self.cur_epoch}")
        return self._finish_current_epoch()

    def reload_settings_and_next_generation(self, *, pocet_aut: int, max_time: float, cars_to_next: int, save_as: str, load_from: str) -> bool:
        if not self.pustene or not self.sprite_list:
            print("RELOAD SET: neni spusteny zadny trening")
            return False

        self.setup(
            pocet_aut=pocet_aut,
            pocet_generaci=self.pocet_generaci,
            max_time=max_time,
            cars_to_next=cars_to_next,
            save_as=save_as,
            load_from=load_from,
        )

        if self.training_logger is not None:
            self.training_logger.save_as = self.save_as
            self.training_logger.load_from = self.load_from

        print(
            "RELOAD SET: nacitam nove nastaveni a ukoncuji aktualni generaci "
            f"(auta={self.pocet_aut}, cars_to_next={self.cars_to_next}, max_time={self.max_time}, "
            f"save_as={self.save_as}, load_from={self.load_from})"
        )
        return self._finish_current_epoch()

    def autosave(self) -> None:
        self.save(self.save_as)

    def save(self, file: str) -> bool:
        self.score_cars()
        if not self.best_cars_list:
            print("SAVE: neni co ulozit, nejdrive spustte nebo nactete trening")
            return False

        print(f"probiha save souboru {file}")
        print(os.getcwd())
        print(self.best_cars_list[0].brain.get_parameters())# nejlepsí je na prvním íste :)
        save_path = Path(PATH_SAVES) / file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **self.best_cars_list[0].brain.get_parameters())
        return True

    def _load_path_candidates(self, file: str) -> list[Path]:
        save_name = str(file or "userbrain.npz").strip() or "userbrain.npz"
        save_names = [save_name]
        if Path(save_name).suffix.lower() != ".npz":
            save_names.append(f"{save_name}.npz")

        candidates = []
        for name in save_names:
            path = Path(name)
            candidate = path if path.is_absolute() else Path(PATH_SAVES) / path
            if candidate not in candidates:
                candidates.append(candidate)

        return candidates

    def load(self, file: str = "userbrain.npz") -> bool:
        self.running = False
        self.pustene = False
        self.paused = False
        self.training_logger = None
        print(f"probiha load souboru {file}")
        candidates = self._load_path_candidates(file)
        params_path = next((path for path in candidates if path.is_file()), None)
        if params_path is None:
            print(f"ERROR: save '{file}' nebyl nalezen.")
            print("Zkousel jsem tyto soubory:")
            for candidate in candidates:
                print(f" - {candidate}")
            return False

        try:
            params =  np.load(params_path)
        except Exception as exc:
            print(f"ERROR: save '{params_path}' se nepodarilo nacist: {exc}")
            return False

        print(f"Nacten save: {params_path}")
        try:
            print(f"Parametry v save: {', '.join(params.files)}")
            test_brain = self.defaultbrain()
            test_brain.set_parameters(params)
        except Exception as exc:
            print(f"ERROR: save '{params_path}' se nepodarilo pouzit pro aktualniho agenta: {exc}")
            return False

        self.reset_brains()

        if len(self.sprite_list) > 0:
            self.sprite_list = list()  # registrujeme do vsech - abych je pak byl schopen prebrat
        if len(self.sprite_running) > 0:
            for c in self.sprite_running:
                c.kill()
            self.sprite_running = pygame.sprite.Group()  # praucjeme jen s running ve kole

        self.cur_epoch = 0
        self.total_time = 0
        self.paused = False
        self.reset_monitoring()
        base_x = TILESIZE * 4 + int(TILESIZE / 2)
        base_y = TILESIZE * 8 + int(TILESIZE / 2)

        # a vytvořím auta:
        try:
            for i in range(self.pocet_aut):
                y = base_y + int(np.random.randint(-SPAWN_Y_JITTER_PX, SPAWN_Y_JITTER_PX + 1))
                c = AI_car(base_x, y, 10, 20, self.brain_list[i],  180+np.random.randint(-45,+45))
                c.brain.set_parameters(params)
                if i>0:
                    c.brain.mutate()
                self.sprite_list.append(c)
                self.sprite_running.add(c)
        except Exception as exc:
            self.sprite_list = list()
            self.sprite_running = pygame.sprite.Group()
            print(f"ERROR: nepodarilo se vytvorit auta ze save '{params_path}': {exc}")
            return False

        # a zapnu tréning
        self._start_training_logger("load")
        self.running = True
        self.pustene = True
        self.paused = False
        return True

    def update(self,  dt: float, keys, blocks) -> None:
        if self.running and not self.paused:
            for c in list(self.sprite_running):
                c.update(dt, keys, blocks)

                if not c.running:
                    self.sprite_running.remove(c)
                    continue

                # nárazy do zdí
                hit = pygame.sprite.spritecollideany(c, blocks.sprites)
                if hit is not None:
                    c.running = False
                    self.sprite_running.remove(c)

            self.total_time += dt


        # enco jako if total time> max time - new epoch
        # ted musím doresit epochy, kolize a save a load
        if self.total_time > self.max_time and self.pustene and not self.paused:
            self._finish_current_epoch()

    def _finish_current_epoch(self) -> bool:
        self.running = False # timer pro epochu
        self.paused = False

        if not self.sprite_list:
            print("ERROR: nelze ukoncit epochu, nejsou vytvorena zadna auta")
            self.pustene = False
            return False

        if self.cur_epoch < self.pocet_generaci:
            print(f"-----epocha {self.cur_epoch}------------------------")
            #for sprite in self.sprite_list:
            #    print(sprite.brain.parameters)
            #print(f"----------------------------------------------------")
            self.setup_next_epoch()
        else:
            self.pustene = False
            self.score_cars()
            self.record_generation_stats()
            if self.training_logger is not None:
                plot_every = getattr(self.training_logger, "plot_every", 0)
                rows_count = len(getattr(self.training_logger, "rows", []))
                if plot_every > 0 and rows_count % plot_every != 0:
                    self.training_logger.save_plot()
            self.autosave()
        return True

    def reset_monitoring(self) -> None:
        self.best_score_all = float("-inf")
        self.stagnation_epochs = 0
        self.completed_generations = 0
        self.last_generation_best_score = 0.0
        self.last_generation_avg_score = 0.0
        self.last_generation_median_score = 0.0
        self.last_generation_crashed_count = 0

    def _score_values(self) -> list[float]:
        return [float(getattr(c.brain, "score", 0.0)) for c in self.sprite_list]

    def _score_summary(self) -> tuple[float, float, float]:
        scores = self._score_values()
        if not scores:
            return 0.0, 0.0, 0.0
        return float(max(scores)), float(np.mean(scores)), float(np.median(scores))

    def record_generation_stats(self) -> None:
        best, avg, median = self._score_summary()
        self.last_generation_best_score = best
        self.last_generation_avg_score = avg
        self.last_generation_median_score = median
        self.last_generation_crashed_count = max(0, len(self.sprite_list) - len(self.sprite_running))
        self.completed_generations += 1

        if best > self.best_score_all:
            self.best_score_all = best
            self.stagnation_epochs = 0
        else:
            self.stagnation_epochs += 1

        if self.training_logger is not None:
            self.training_logger.log_generation(self.get_training_stats())

    def get_training_stats(self) -> dict[str, Any]:
        best, avg, median = self._score_summary()
        best_all = best if self.best_score_all == float("-inf") else max(self.best_score_all, best)
        total = len(self.sprite_list)
        active = len(self.sprite_running)
        crashed = max(0, total - active)

        return {
            "running": self.running,
            "training_active": self.pustene,
            "paused": self.paused,
            "generation": self.cur_epoch,
            "generations_total": self.pocet_generaci,
            "completed_generations": self.completed_generations,
            "time": self.total_time,
            "max_time": self.max_time,
            "cars_total": total,
            "cars_active": active,
            "cars_crashed": crashed,
            "best_score": best,
            "best_score_all": best_all,
            "avg_score": avg,
            "median_score": median,
            "last_generation_best_score": self.last_generation_best_score,
            "last_generation_avg_score": self.last_generation_avg_score,
            "last_generation_median_score": self.last_generation_median_score,
            "last_generation_crashed_count": self.last_generation_crashed_count,
            "stagnation_epochs": self.stagnation_epochs,
            "save_as": self.save_as,
        }


        #print(f"total time: {self.total_time}")
        #print(f"setting, pocet aut: {self.pocet_aut}, pocet generaci: {self.pocet_generaci}, epoch: {self.epoch}")
        #print(f"vnitrni data: len sprite list: {len(self.sprite_list)}, len sprite running: {len(self.sprite_running)}")
