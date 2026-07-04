# AI Cart V2

AI Cart V2 je výukový Pygame projekt pro evoluční trénování závodních AI agentů. Studenti píšou vlastní "mozek" auta, trénují ho na mapě, ukládají parametry do `.npz` souboru a potom mohou agenty porovnávat v duelu nebo benchmarku.

Autor: Karel Šafr, PhD

## Pro koho projekt je

- studenti, kteří chtějí vytvořit vlastního agenta
- vyučující, kteří chtějí spouštět tréninky, benchmarky a porovnání týmů
- kdokoliv, kdo si chce prohlédnout jednoduchý evoluční závodní engine v Pythonu

## Rychlé spuštění

Spouštějte projekt z kořene `AI_Cart_engine_V2/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy
python main.py
```

Na Windows použijte aktivaci virtuálního prostředí podle svého terminálu, například `.venv\Scripts\activate`.

## Kde co najít

- [STUDENT_GUIDE.md](STUDENT_GUIDE.md) - hlavní návod pro studenty: vytvoření agenta, trénink, save/load, mapy, duel a benchmark
- [AGENT_INTERFACE.md](AGENT_INTERFACE.md) - přesný kontrakt mezi enginem a studentským agentem
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - mapa projektu, běhový tok a hranice mezi enginem a studentskou prací
- [students/README.md](students/README.md) - krátký tahák přímo ve studentské složce

## Základní princip

Student dodá hlavně dva soubory:

```text
students/agents/AIbrain_<NazevTymu>.py
students/saves/AIbrain_<NazevTymu>.npz
```

Agent se načítá automaticky ze `students/agents/`. Natrénované parametry se ukládají do `students/saves/`. Vlastní mapy patří do `students/maps/` a výsledky benchmarků, tréninkové CSV logy a grafy do `students/results/`.

Výchozí agent je `AIbrain_linear`.

## První orientace

Pokud jste student a chcete začít pracovat, otevřete [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

Pokud chcete jen vědět, jaké metody musí agent mít, otevřete [AGENT_INTERFACE.md](AGENT_INTERFACE.md).

Pokud upravujete engine nebo kontrolujete strukturu projektu, otevřete [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
