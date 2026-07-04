# AI Cart V2 - návod pro studenty

AI Cart V2 je jednoduchý Pygame závodní engine pro evoluční trénování AI agentů. Student píše vlastní "mozek" auta, trénuje ho na mapě, uloží parametry do `.npz` souboru a potom se agent může porovnávat v duelu nebo benchmarku.

Engine a studentská práce jsou oddělené. Studenti běžně pracují jen ve složce `students/`.

Hlavní rozcestník projektu je v `README.md`. Přesný kontrakt rozhraní agenta je v `AGENT_INTERFACE.md`.

## Rychlý start

Spouštějte projekt z kořene `AI_Cart_engine_V2/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy
python main.py
```

Na Windows použijte aktivaci virtuálního prostředí podle svého terminálu, například `.venv\Scripts\activate`.

## Co odevzdává tým

Každý tým odevzdává hlavně tyto dva soubory:

```text
students/agents/AIbrain_<NazevTymu>.py
students/saves/AIbrain_<NazevTymu>.npz
```

První soubor je zdrojový kód agenta. Druhý soubor jsou natrénované parametry uložené z tréninku.

Příklad:

```text
students/agents/AIbrain_SAFR.py
students/saves/AIbrain_SAFR.npz
```

Soubor agenta a třída uvnitř musí mít stejné jméno:

```python
class AIbrain_SAFR:
    ...
```

## Co studenti smí měnit

Studenti mohou přidávat a upravovat:

- `students/agents/AIbrain_<NazevTymu>.py`
- `students/saves/*.npz`
- `students/maps/*.csv`
- případně `students/benchmark_agents.yaml`, pokud připravují vlastní benchmark

Studenti by neměli měnit engine:

- `core/`
- `scenes/`
- `my_sprites/`
- `UI/`
- `main.py`
- `constants.py`
- `AI_engines/`

`AI_engines/` slouží jako vestavěné ukázky a šablony, ne jako místo pro studentské odevzdání.

## Struktura studentské práce

```text
students/
├── agents/      # Python soubory agentů
├── saves/       # uložené .npz parametry
├── maps/        # vlastní .csv mapy
├── results/     # benchmarky, logy tréninku a grafy
└── benchmark_agents.yaml
```

Agenti se načítají automaticky ze dvou míst:

1. `students/agents/`
2. `AI_engines/`

Pokud existuje agent se stejným názvem v obou složkách, přednost má verze ve `students/agents/`.

Výchozí agent po spuštění je `AIbrain_linear`.

## Jak má vypadat agent

Agent je Python třída, kterou auto používá pro rozhodování. Vstupem do rozhodování jsou vzdálenosti paprsků od překážek. Výstupem jsou čtyři hodnoty:

```text
index 0 - plyn
index 1 - brzda
index 2 - doleva
index 3 - doprava
```

Auto bere akci jako aktivní, pokud je hodnota větší než `0.5`.

Agent musí mít stejné rozhraní jako ukázkové agenty. Nejlepší výchozí vzor je:

```text
AI_engines/AIbrain_linear.py
```

Přesný kontrakt mezi enginem a agentem je v:

```text
AGENT_INTERFACE.md
```

Povinné metody:

- `__init__(self)`
- `decide(self, data)`
- `mutate(self)`
- `store(self)`
- `get_parameters(self)`
- `set_parameters(self, parameters)`
- `calculate_score(self, distance, time, no)`
- `passcardata(self, x, y, speed)`
- `getscore(self)`

Praktické minimum:

- v `__init__` inicializujte parametry, `self.score` a `self.NAME`
- v `decide` vraťte 4 čísla pro ovládání auta
- v `mutate` náhodně upravte parametry
- ve `store` uložte parametry do `self.parameters`
- v `set_parameters` obnovte stav ze savu
- v `calculate_score` nastavujte fitness do `self.score`

## Hlavní menu

Po spuštění `python main.py` se otevře hlavní menu:

- `Hraj` - ruční jízda jedním autem
- `Mapa` - editor map
- `Trénuj` - evoluční trénink vybraného agenta
- `Souboj` - duel dvou agentů
- `Benchmark` - hromadné porovnání více agentů
- `Konec` - ukončení aplikace

Ovládání:

- šipky nahoru/dolů - výběr položky
- Enter - potvrzení
- Esc - návrat nebo ukončení

Ve spodní části menu jsou dvě textová pole:

- `agent_name` - název agenta, například `AIbrain_SAFR`
- mapa - název mapy, například `DefaultRace`, `DefaultReset` nebo název ze `students/maps/`

Názvy map se zadávají bez cesty. Přípona `.csv` je volitelná.

## Trénink agenta

Postup:

1. Vložte soubor agenta do `students/agents/`.
2. Spusťte `python main.py`.
3. V menu do `agent_name` napište název agenta, například `AIbrain_SAFR`.
4. Do pole mapy napište například `DefaultRace`.
5. Vyberte `Trénuj`.
6. Nastavte parametry tréninku.
7. Stiskněte `START`.

Pole v tréninku:

- `pocet_aut` - počet aut v jedné generaci
- `pocet_generaci` - počet generací
- `cars_to_next` - kolik nejlepších aut se použije pro další generaci; musí být mezi `1` a `pocet_aut`
- `save_as` - název savu pro nejlepšího agenta
- `max_time` - maximální délka jedné generace v sekundách
- `load_from` - save, ze kterého se má pokračovat

Tlačítka:

- `START` - spustí trénink od začátku s vybraným agentem
- `PAUSE` / `RESUME` - pozastaví nebo znovu spustí aktuální běh bez ztráty generace
- `NEXT GEN` - ručně ukončí aktuální generaci a vybere nejlepší auta stejně jako při vypršení času
- `RELOAD SET` - načte z polí nové `pocet_aut`, `cars_to_next`, `save_as`, `load_from` a `max_time`, potom ukončí aktuální generaci stejnou logikou jako `NEXT GEN`
- `RESET` - zastaví a vymaže aktuální běh tréninku
- `SAVE` - uloží aktuálně nejlepší mozek podle pole `save_as`
- `LOAD` - načte save z `load_from` a spustí trénink z načtených parametrů

`RELOAD SET` nepouští `LOAD` ze souboru. Jen přebere nové hodnoty nastavení pro další generaci a zachová běžnou logiku výběru nejlepších aut a mutací.

Pokud je `cars_to_next` mimo povolený rozsah, program ho upraví na nejbližší bezpečnou hodnotu a vypíše varování do konzole. `SAVE` před spuštěním nebo načtením tréninku jen vypíše hlášku, protože ještě není co ukládat.

Doporučení pro názvy savů:

```text
AIbrain_<NazevTymu>.npz
```

Například:

```text
AIbrain_SAFR.npz
```

Savy se ukládají do:

```text
students/saves/
```

## Načítání savů

Tlačítko `LOAD` v tréninku hledá save v tomto pořadí:

```text
students/saves/<nazev>
students/saves/<nazev>.npz
```

To znamená, že do `load_from` můžete napsat například `FAST` i `FAST.npz`.

Pokud save neexistuje, nejde otevřít nebo nepasuje k aktuálně vybranému agentovi, program nespadne. Vypíše chybu do konzole a trénink nespustí.

Důležité: save musí odpovídat architektuře agenta. Například parametry pro `AIbrain_2layer` nemusí jít načíst do `AIbrain_linear`.

## Monitoring tréninku

Vpravo v tréninkové scéně je jednoduchý živý monitor:

- `Agent` - aktuálně vybraný agent
- `Gen` - aktuální generace a čas generace
- `Alive` - kolik aut ještě jede
- `Crash` - kolik aut skončilo
- `Best` - nejlepší skóre v aktuální generaci
- `All` - nejlepší skóre za celý běh
- `Avg` - průměrné skóre generace
- `Med` - medián skóre generace
- `Stag` - kolik generací se nezlepšilo nejlepší skóre
- `Save` - cílový název savu

Monitor je orientační. Když rychle roste `Crash` a `Best` se nezlepšuje, agent nejspíš potřebuje lepší rozhodování, mutaci nebo skórování.

## Logy a grafy tréninku

Každý nový `START` nebo úspěšný `LOAD` v tréninku založí vlastní CSV log. Po každé dokončené generaci se přidá jeden řádek s metrikami:

- nejlepší skóre generace
- průměrné skóre generace
- medián skóre generace
- počet naražených aut
- počet aktivních aut
- nejlepší skóre celého běhu
- stagnace, čas generace, agent, mapa a název savu

CSV logy se ukládají do:

```text
students/results/training_logs/
```

Pokud je dostupný balíček `matplotlib`, trénink navíc průběžně ukládá PNG graf do:

```text
students/results/training_plots/
```

Výchozí interval grafu je každých 5 dokončených generací. Nastavení je v `constants.py`:

```python
TRAINING_PLOT_EVERY = 5
```

Hodnota `10` znamená ukládání každých 10 generací. Hodnota `0` PNG graf vypne. Pokud `matplotlib` není nainstalovaný nebo graf nejde uložit, program jen vypíše hlášku do konzole a trénink pokračuje dál. CSV logování na `matplotlib` nezávisí.

## Mapy

Výchozí mapy jsou v:

```text
DefaultSettings/
```

Studentské mapy jsou v:

```text
students/maps/
```

Použitelné názvy v menu:

- `DefaultRace` - výchozí závodní trať
- `DefaultReset` - základní mapa pro editor
- vlastní název mapy ze `students/maps/`, například `moje_mapa`

Editor map:

- v menu vyberte `Mapa`
- do textového pole napište název mapy bez `.csv`
- `Save map` uloží mapu do `students/maps/<nazev>.csv`
- `Load` načte mapu ze `students/maps/<nazev>.csv`
- `RESET` načte `DefaultSettings/DefaultReset.csv`

Startovní tile je:

```text
road_dirt42
```

Klikáním do levé části mapy měníte dlaždice. Opakované kliknutí na stejnou pozici cykluje dostupné typy dlaždic.

## Souboj dvou agentů

V menu vyberte `Souboj`.

Pole vpravo:

- `map_name` - název mapy
- `engine1_class` - třída prvního agenta
- `engine1_save` - save prvního agenta
- `engine2_class` - třída druhého agenta
- `engine2_save` - save druhého agenta

Tlačítka:

- `LOAD MAP` - načte mapu
- `START DUEL` - spustí duel

Agent se hledá podle třídy/souboru. Save se hledá v `students/saves/`.

## Benchmark více agentů

Benchmark slouží k porovnání více agentů najednou. V menu vyberte `Benchmark`.

Výchozí soubor se seznamem agentů:

```text
students/benchmark_agents.yaml
```

Formát:

```yaml
agents:
  - agent: AIbrain_TeamA
    save: AIbrain_TeamA.npz
  - agent: AIbrain_TeamB
    save: AIbrain_TeamB.npz
```

Pole v benchmark scéně:

- `map_name` - mapa pro benchmark
- `finish_row` - řádek cílové dlaždice
- `finish_col` - sloupec cílové dlaždice
- `time_limit_s` - časový limit
- `output_csv` - název výstupního CSV
- `agents_file` - YAML nebo TXT soubor se seznamem agentů
- `engines_fallback` - ruční seznam agentů, pokud soubor není dostupný

Výchozí `agents_file` je `benchmark_agents.yaml`, což se překládá na:

```text
students/benchmark_agents.yaml
```

Výstupy benchmarku se ukládají do:

```text
students/results/
```

Během běžícího benchmarku se vlevo na mapě zobrazuje bílý seznam `#pořadí název_mozku`. Pořadí odpovídá YAML/TXT seznamu. Stejné číslo se kreslí výrazně žlutě s tmavým obrysem přímo přes auto a ukládá se i do výsledkového CSV ve sloupci `benchmark_order`.

Výsledný `rank` je závodní pořadí: auta, která dojela do cíle, jsou řazená podle času. Auta, která nedojela, jsou až za nimi podle stavu `TIMEOUT`, `CRASH`, `OUT`. Najetá vzdálenost `distance_tiles` zůstává jen informační metrika v CSV a ve výsledkovém výpisu.

Ruční fallback formát:

```text
AIbrain_TeamA:AIbrain_TeamA.npz; AIbrain_TeamB:AIbrain_TeamB.npz
```

Podporované oddělovače položek jsou `;` nebo `|`. Mezi agentem a savem lze použít `:`, `,` nebo mezeru.

## Doporučený studentský postup

1. Zkopírujte si myšlenkově vzor z `AI_engines/AIbrain_linear.py`.
2. Vytvořte `students/agents/AIbrain_<NazevTymu>.py`.
3. Zachovejte povinné metody a stejný název souboru i třídy.
4. V menu vyberte svého agenta.
5. Trénujte na `DefaultRace` a na vlastních mapách.
6. Sledujte monitor tréninku.
7. Uložte nejlepší save do `students/saves/AIbrain_<NazevTymu>.npz`.
8. Ověřte, že `LOAD` dokáže váš save načíst.
9. Vyzkoušejte benchmark nebo duel.
10. Odevzdejte `.py` a `.npz`.

## Časté problémy

`Agent nebyl nalezen`

- zkontrolujte, že soubor je ve `students/agents/`
- zkontrolujte, že název souboru a třídy je stejný
- zkontrolujte, že třída implementuje povinné metody

`Save nebyl nalezen`

- zkontrolujte `students/saves/`
- do `load_from` můžete zadat název s `.npz` i bez `.npz`

`Save nejde použít pro aktuálního agenta`

- nejspíš načítáte parametry jiné architektury
- vyberte správný agent v menu nebo správný `.npz` soubor

`Mapa neexistuje`

- pro výchozí mapy použijte `DefaultRace` nebo `DefaultReset`
- pro vlastní mapy zkontrolujte `students/maps/<nazev>.csv`

`Benchmark nenašel agenty`

- zkontrolujte `students/benchmark_agents.yaml`
- zkontrolujte odsazení YAML
- zkontrolujte, že agent existuje a save je ve `students/saves/`

## Typování kódu

Engine používá postupné, měkké typování. Sdílené protokoly jsou v:

```text
core/protocols.py
```

Nejdůležitější je `AgentProtocol`, který popisuje metody, které musí studentský agent nabídnout. Typování slouží pro lepší orientaci, editor a bezpečnější úpravy enginu; nemění běh programu.

Python soubory v projektu používají `from __future__ import annotations`, takže lze anotace doplňovat postupně a bez vlivu na chování programu.

Konfigurace pro `pyright` a `mypy` je v `pyproject.toml`. Studentské agenty nejsou v základní kontrole striktně vynucované, aby zůstaly flexibilní pro výuku. Pokud máte nástroj nainstalovaný, lze kontrolu spustit například:

```bash
pyright
mypy
```

## Poznámka k zvuku

V `main.py` je nastavený dummy audio driver:

```python
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
```

Je to kvůli stabilitě Pygame na některých systémech. Pro práci studentů není potřeba nic měnit.
