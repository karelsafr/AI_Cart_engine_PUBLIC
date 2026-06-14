AI engines training – návod pro studenty

Tento projekt je jednoduchá závodní hra v Pygame, která slouží jako „testovací polygon“ pro trénování vlastních AI agentů (mozků) řízení auta.
Vaším úkolem je navrhnout, naimplementovat a natrénovat vlastní mozek tak, aby se auto naučilo bezpečně a rychle projet trať.

------------------------------------------------------------
0. Balíčky
------------------------------------------------------------
Engine funguje v aktuální chvíli primárně na pygame-ce (importuje se stejně jen je nutné vyměnit knihovnu). 
Dále numpy, pandas.

------------------------------------------------------------
1. Co máte odevzdat
------------------------------------------------------------

Každý tým odevzdá dva soubory se stejným základním jménem:

1) Zdrojový kód mozku
   - Soubor: AIbrain_/AIbrain_<NazevTymu>.py
   - Třída uvnitř: AIbrain_<NazevTymu>
   - Příklad: AI_engines/AIbrain_SAFR.py s třídou AIbrain_SAFR

2) Natrénované parametry mozku
   - Soubor: UserData/SAVES/AIbrain_<NazevTymu>.npz
   - Vznikne uložením z tréninkové scény (viz níže, pole save_as)
   - Příklad: UserData/SAVES/AIbrain_SAFR.npz

Oba soubory (.py i .npz) pak odevzdejte dle instrukcí vyučujícího.
Na základě těchto souborů se následně spustí souboje mozků (Duel) mezi jednotlivými týmy.

------------------------------------------------------------
2. Co smíte a nesmíte v kódu měnit
------------------------------------------------------------

Smíte:

1) Přidat svůj mozek do složky AI_engines/
   - Vytvořit nový soubor AIbrain_<NazevTymu>.py na základě vzorů:
     - AI_engines/AIbrain_linear.py – jednoduchý lineární mozek
     - AI_engines/AIbrain_TeamName.py – demonstrační šablona
2) Upravit jediný import v main.py v sekci „Students part“ – tj. vybrat, který mozek se má používat při tréninku:


   Výchozí (ukázkový) mozek:

   from AI_engines.AIbrain_linear import AIbrain_linear as trainbrain


   Pro váš tým to má vypadat např. takto:

   from AI_engines.AIbrain_SAFR import AIbrain_SAFR as trainbrain

Nesmíte:

- Měnit žádné jiné moduly ani soubory hry (scény, sprity, správu aut, duel atd.).
- Měnit rozhraní tříd AI_car, Car_manager, SceneManager atd.
- Měnit strukturu složek, názvy souborů mimo svého mozku a svých .npz uložených v UserData/SAVES/.

Jediné dvě oblasti, kde píšete svůj kód, jsou:

- nový soubor AI_engines/AIbrain_<NazevTymu>.py
- úprava jednoho importu v main.py v sekci „### Students part“

------------------------------------------------------------
3. Jak vypadá „mozek“ (AI brain)
------------------------------------------------------------

Mozek je Python třída, kterou hra používá k rozhodování o ovládání auta. Vstupy jsou zejména vzdálenosti paprsků (raycast) od překážek,
výstupy jsou čtyři „akce“:

- index 0 – dopředu (plyn)
- index 1 – dozadu (brzda)
- index 2 – doleva (zatáčení)
- index 3 – doprava

Minimální požadavky na třídu mozku:

- Třída se jmenuje AIbrain_<NazevTymu>.
- Je v souboru AIbrain_<NazevTymu>.py ve složce AI_engines/.
- Implementuje stejné rozhraní jako AIbrain_linear, zejména metody:

  - __init__(self) + init_param(self)
    - vytvoření a inicializace parametrů modelu
    - nastavení self.NAME
    - volání self.store()

  - decide(self, data)
    - přijímá seznam/vektor vzdáleností paprsků
    - vrací 4 čísla (pro akce [up, down, left, right])
    - hra pak používá práh > 0.5

  - mutate(self)
    - náhodně pozmění parametry modelu (nutné pro evoluční trénink)

  - store(self)
    - uloží všechny parametry do slovníku self.parameters

  - get_parameters(self) a set_parameters(self, parameters)
    - ukládání/načítání parametrů (používá se při SAVE/LOAD a v Duelu)

  - calculate_score(self, distance, time, no)
    - aktualizace self.score (fitness)

  - passcardata(self, x, y, speed)
    - hra ji volá automaticky
    - můžete využít ve skórování nebo ignorovat

  - getscore(self)
    - vrací aktuální hodnotu self.score

Doporučení:

Jako nejčistší vzor použijte AI_engines/AIbrain_linear.py.
Stačí upravit logiku v init_param, decide a mutate podle vašich nápadů a zachovat stejné rozhraní metod.

------------------------------------------------------------
4. Jak projekt spustit
------------------------------------------------------------

1) Nainstalujte Python 3.12 a více
   Projekt je testován s běžnou verzí Pythonu 3 (3.12+).

2) (Doporučeno) Vytvořte virtuální prostředí


3) Nainstalujte potřebné balíčky: numpy a pandas, např:

   pip install pygame numpy

4) Spusťte hru:

   python main.py

Po spuštění se objeví hlavní menu.

------------------------------------------------------------
5. Ovládání menu a herní módy
------------------------------------------------------------

V hlavním menu najdete položky:

- Hraj   – ruční řízení jednoho auta klávesnicí (pro pochopení fyziky a mapy)
- Mapa   – editor map (tvorba/úprava tratí)
- Trénuj – evoluční trénink vašeho mozku
- Souboj – duel dvou uložených mozků
- Konec  – ukončení aplikace

Ovládání menu:

- Šipky nahoru/dolů – výběr položky
- Enter             – potvrzení výběru
- Esc               – návrat / ukončení

V dolní části menu je textové pole pro název mapy.
Používají se názvy:

- DefaultRace – výchozí závodní trať
- DefaultReset – prázdná mapa / základní nastavení
- Nebo název mapy, kterou si sami uložíte v editoru (UserData/<nazev>.csv)

------------------------------------------------------------
6. Jak natrénovat vlastní mozek
------------------------------------------------------------

1) Ujistěte se, že v main.py používáte svůj mozek:

   from AI_engines.AIbrain_MujTeam import AIbrain_MujTeam as trainbrain

2) Spusťte hru (python main.py) a v menu zvolte mapu (např. DefaultRace) v textovém poli.

3) V menu vyberte položku „Trénuj“. Otevře se tréninková scéna. Vpravo uvidíte textová pole:

   - pocet_aut     – kolik aut (jedinců) je v jedné generaci
   - pocet_generaci – kolik generací se má odsimulovat
   - cars_to_next  – kolik nejlepších aut se bere jako „elitní“ pro další generaci
   - save_as       – název souboru, do kterého se uloží NEJLEPŠÍ mozek
                      (zde nastavte AIbrain_<NazevTymu>.npz, bez cesty, uloží se do UserData/SAVES/)
   - max_time      – maximální délka jedné generace v sekundách
   - load_from     – název souboru .npz (z UserData/SAVES/), ze kterého chcete pokračovat v tréninku

4) Tlačítka ve tréninkové scéně:

   - START
     - spustí trénink
     - vytvoří populaci aut s vaším mozkem
     - spustí evoluční cyklus (běh, skórování, výběr, mutace)

   - STOP
     - zastaví probíhající trénink

   - SAVE AS
     - uloží aktuálně nejlepší mozek do souboru uvedeného v save_as

   - LOAD & PLAY
     - načte mozek z load_from
     - nastaví ho jako výchozí
     - spustí trénink s tímto startovním mozkem (plus mutace)

5) Doporučený postup pro odevzdání:

   - Trénujte, dokud nejste s výsledkem spokojeni.
   - nezapomente ze můzete vytvářet vlastní mapy. Nevíte na jakých mapách hru pustím!
   - Nastavte save_as na:

     AIbrain_<NazevTymu>.npz

   - Stiskněte SAVE AS.
   - Zkontrolujte, že se soubor objevil v UserData/SAVES/.

------------------------------------------------------------
7. Jak probíhá souboj mozků (Duel)
------------------------------------------------------------

Souboje mozků mezi týmy probíhají v záložce „Souboj“:

- Vpravo jsou textová pole:
  - map_name      – název mapy (např. DefaultRace)
  - engine1_class – název třídy mozku 1 (např. AIbrain_TeamA)
  - engine1_save  – název .npz souboru z UserData/SAVES/ pro mozek 1 (např. AIbrain_TeamA.npz)
  - engine2_class – název třídy mozku 2 (např. AIbrain_TeamB)
  - engine2_save  – název .npz souboru z UserData/SAVES/ pro mozek 2 (např. AIbrain_TeamB.npz)

- Tlačítka:
  - LOAD MAP   – načte mapu
  - START DUEL – spustí souboj dvou aut s danými mozky

Mozky se dynamicky importují podle jména třídy a souboru .npz.
To je důvod, proč je důležité, aby:

- soubor s mozkem měl tvar AIbrain_<NazevTymu>.py
- třída uvnitř měla přesně stejné jméno
- soubor s parametry měl sjednocené jméno AIbrain_<NazevTymu>.npz a ležel v UserData/SAVES/

------------------------------------------------------------
8. Map editor – tvorba vlastní trati
------------------------------------------------------------

V menu vyberte položku „Mapa“:

- Vpravo je textové pole pro název mapy (bez přípony).
- Tlačítka:
  - Save map – uloží mapu jako UserData/<nazev>.csv
  - Load     – načte mapu UserData/<nazev>.csv
  - RESET    – načte výchozí šablonu mapy z DefaultSettings/DefaultReset.csv

Do mapy kreslíte klikáním do levé části okna.
Opakované klikání na stejný tile cykluje mezi typy dlaždic.
Startovní tile pro závody je road_dirt42.

Mapy si můžete vytvořit pro ladění vlastního mozku.
Pro oficiální vyhodnocení se může použít konkrétní vybraná trať (viz zadání od vyučujícího).

------------------------------------------------------------
9. Poznámky k prostředí a kódu
------------------------------------------------------------

- Velikost okna a dlaždic se nastavuje v souboru constants.py:
  - parametr HEIGHT určuje výšku okna
  - ostatní rozměry (WIDTH, TILESIZE, velikosti menu atd.) se dopočítávají automaticky

- Audio na Linuxu:
  Na Linuxu může Pygame mixer způsobovat významné lagy (0–15 s, někdy i 1–2 min).
  Proto je v main.py použito:

  os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

  To vypne zvuk a používá „dummy“ driver (bez zvukového výstupu).
  Na Windows a macOS není potřeba nic měnit (nic není treba nikde měnit ;) )

------------------------------------------------------------
10. Shrnutí
------------------------------------------------------------

1) Vytvořte soubor AI_engines/AIbrain_<NazevTymu>.py a v něm třídu AIbrain_<NazevTymu>
   se stejným rozhraním jako AIbrain_linear.

2) V main.py v sekci „### Students part“ nastavte import svého mozku jako trainbrain.

3) Spusťte hru, v módu „Trénuj“ natrénujte mozek a uložte jej jako AIbrain_<NazevTymu>.npz do UserData/SAVES/.

4) Na Git odevzdejte:
   - AI_engines/AIbrain_<NazevTymu>.py
   - UserData/SAVES/AIbrain_<NazevTymu>.npz

5) zbytek (souboje mozků v módu „Souboj“) se postaráme my.