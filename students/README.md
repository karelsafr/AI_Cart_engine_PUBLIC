# Students workspace

Tato složka je určená pro studentské agenty, savy, mapy a benchmark výstupy. Engine je mimo tuto složku.

Podrobný studentský návod je v `../STUDENT_GUIDE.md`. Přesný kontrakt agenta je v `../AGENT_INTERFACE.md`.

```text
students/
├── agents/                 # AIbrain_<NazevTymu>.py
├── saves/                  # AIbrain_<NazevTymu>.npz
├── maps/                   # vlastní .csv mapy
├── results/                # benchmarky, logy tréninku a grafy
├── benchmark_agents.yaml   # seznam agentů pro benchmark
└── README.md
```

## Agent

Soubor i třída musí mít stejné jméno:

```text
students/agents/AIbrain_MujTeam.py
class AIbrain_MujTeam
```

Jako vzor použijte:

```text
AI_engines/AIbrain_linear.py
```

Přesný kontrakt rozhraní agenta je v:

```text
../AGENT_INTERFACE.md
```

Agent musí implementovat metody:

- `decide`
- `mutate`
- `store`
- `get_parameters`
- `set_parameters`
- `calculate_score`
- `passcardata`
- `getscore`

## Save

Trénink ukládá `.npz` soubory do:

```text
students/saves/
```

Doporučený název:

```text
AIbrain_<NazevTymu>.npz
```

Při načítání můžete zadat název s `.npz` i bez `.npz`.

V tréninku musí být `cars_to_next` mezi `1` a `pocet_aut`. Pokud zadáte hodnotu mimo rozsah, engine ji upraví na bezpečnou hodnotu a vypíše varování. `SAVE` před spuštěním nebo načtením tréninku jen vypíše hlášku, protože zatím není co uložit.

Tlačítko `RELOAD SET` během běžícího tréninku přebere nové hodnoty `pocet_aut`, `cars_to_next`, `save_as`, `load_from` a `max_time` a vytvoří další generaci stejnou logikou jako `NEXT GEN`. Save ze souboru tím nenačítá.

## Logy tréninku

Trénink po každé dokončené generaci automaticky zapisuje CSV řádek do:

```text
students/results/training_logs/
```

Pokud je dostupný `matplotlib`, průběžný PNG graf se ukládá do:

```text
students/results/training_plots/
```

CSV funguje vždy samostatně. Když `matplotlib` chybí, trénink pokračuje bez grafu.

## Mapy

Vlastní mapy ukládejte přes editor do:

```text
students/maps/<nazev>.csv
```

V menu pak stačí zadat jen `<nazev>`.

## Benchmark

Seznam agentů pro benchmark je v:

```text
students/benchmark_agents.yaml
```

Formát:

```yaml
agents:
  - agent: AIbrain_MujTeam
    save: AIbrain_MujTeam.npz
```

Výsledky benchmarku se ukládají do:

```text
students/results/
```

V běžícím benchmarku je vlevo na mapě bílý seznam `#pořadí název_mozku`. Stejné pořadové číslo je výrazně žlutě s tmavým obrysem vykreslené přímo přes auto a ve výsledkovém CSV jako `benchmark_order`.

Výsledný `rank` zvýhodňuje skutečné dojetí do cíle: dojetá auta se řadí podle času, nedojetá auta jsou až za nimi podle stavu `TIMEOUT`, `CRASH`, `OUT`. `distance_tiles` je pouze doplňková informace.

## Odevzdání

Typicky odevzdáváte:

```text
students/agents/AIbrain_<NazevTymu>.py
students/saves/AIbrain_<NazevTymu>.npz
```

Neměňte soubory enginu mimo `students/`, pokud k tomu není výslovné zadání.

## Poznámka k typování

Engine má typový protokol `AgentProtocol`, který popisuje povinné metody agenta. Pro vás je pořád nejdůležitější zachovat stejné metody jako v `AI_engines/AIbrain_linear.py`; typování jen pomáhá editoru a kontrole enginu. Studentské agenty mohou anotace používat, ale nejsou kvůli výuce striktně vynucené.
