# Catan AI - Projekt gry w Osadników z Catanu z agentami Deep Q-Learning

## Opis plików

### Główne pliki projektu
- **play.py** - Uruchamia rozgrywkę między dwoma agentami AI z wizualizacją planszy
- **visualizer.py** - Zawiera klasę `CatanVisualizer` do graficznej wizualizacji planszy Catan


### Agenci AI
- **base_agent.py** - Abstrakcyjna klasa bazowa dla wszystkich agentów AI
- **early_agent.py** - Agent do fazy początkowej gry (wybór pierwszych osad), używa DQN z 12 cechami wejściowymi
- **mid_agent.py** - Agent do fazy środkowej gry (budowa, handel), używa DQN z 19 cechami wejściowymi i 5 akcjami wyjściowymi

### Skrypty treningowe
- **train_early.py** - Trenuje model `EarlyAgent` do wyboru optymalnych pozycji startowych
- **train_mid.py** - Trenuje model `MidAgent` do podejmowania decyzji w trakcie gry

### Pozostałe
- **__init__.py** - Plik inicjalizacyjny pakietu (wymaga modułów `core.game` i `core.utils`, których brakuje w tym zestawie)

---

## System nagradzania modeli

### Early Agent (wybór pozycji startowych)

Model otrzymuje **12 cech** dla każdego wierzchołka:

**Cechy wejściowe:**
1. **resource_pips["drewno"]** - suma pips dla drewna (0-15)
2. **resource_pips["cegla"]** - suma pips dla cegły (0-15)
3. **resource_pips["owca"]** - suma pips dla owcy (0-15)
4. **resource_pips["zboze"]** - suma pips dla zboża (0-15)
5. **resource_pips["ruda"]** - suma pips dla rudy (0-15)
6. **len(resources)** - liczba różnych typów surowców (0-5)
7. **has_wood_brick** - czy ma dostęp do drewna i cegły (0/1)
8. **has_wheat_ore** - czy ma dostęp do zboża i rudy (0/1)
9. **has_sheep_wheat** - czy ma dostęp do owcy i zboża (0/1)
10. **is_on_edge** - czy wierzchołek jest na brzegu planszy (0/1)
11. **total_pips** - suma wszystkich pips (0-75)
12. **has_all_resources** - czy ma dostęp do wszystkich 5 surowców (0/1)

**Funkcja nagrody:**
```python
def compute_reward(state):
    production = sum(state[:5])              # Suma pips dla surowców
    diversity = state[5] * 1.5               # Liczba różnych surowców * 1.5
    synergy = state[6] * 3 + state[7] * 3 + state[8] * 2  # Bonusy za pary surowców
    edge_penalty = -2 if state[9] == 1 else 0  # Kara za brzeg
    total_bonus = state[10] * 0.1            # Bonus za ogólną produktywność
    all_resources_bonus = state[11] * 5      # Duży bonus za wszystkie 5 surowców
    
    return production + diversity + synergy + edge_penalty + total_bonus + all_resources_bonus
```

**Nagrody:**
- **Produkcja** - suma wartości pips (0-5) dla każdego surowca dostępnego na wybranym wierzchołku
- **Różnorodność** - +1.5 punktu za każdy unikalny typ surowca (max 7.5 za 5 typów)
- **Synergia drewno+cegła** - +3 punkty (potrzebne na drogi)
- **Synergia zboże+ruda** - +3 punkty (potrzebne na miasta)
- **Synergia owca+zboże** - +2 punkty (potrzebne na karty rozwoju)
- **Kara za brzeg** - -2 punkty jeśli wierzchołek graniczy z mniej niż 3 hexami
- **Bonus za ogólną produktywność** - +0.1 × suma wszystkich pips
- **Bonus za wszystkie surowce** - +5 punktów jeśli ma dostęp do wszystkich 5 typów

---

### Mid Agent (gra właściwa)

Model wybiera akcję (0-4) i otrzymuje nagrody:

**Akcje:**
- `0` - Nic nie rób: **-2 punkty** (zniechęca do bezczynności)
- `1` - Buduj drogę: **+10** jeśli udane, **-9** jeśli niemożliwe
- `2` - Buduj osadę: **+200** jeśli udane, **-20** jeśli niemożliwe
- `3` - Buduj miasto: **+200** jeśli udane, **-15** jeśli niemożliwe
- `4` - Handel z bankiem: **+5** jeśli udany, **-3** jeśli niemożliwy

**Dodatkowe nagrody:**
- **Zasoby z rzutu kostką** - różnica w liczbie surowców przed i po rzucie (zazwyczaj 0-6)
- **Wygrana** - **+99999** punktów przy osiągnięciu 10 punktów zwycięstwa

**Parametry treningu:**
- Gamma (współczynnik dyskontowania): 0.95
- Learning rate: 0.0005
- Epsilon (eksploracja): 1.0 → 0.05 (decay 0.995)
- Liczba tur na grę: 50
- Liczba epizodów: 2000

---

## Struktura katalogów (wymagana, ale niekompletna w zestawie)

Projekt wymaga następującej struktury:
```
projekt/
├── core/
│   ├── __init__.py
│   ├── game.py          # logika gry Catan
│   ├── utils.py         # stałe i funkcje pomocnicze
│   └── visualizer.py    # wizualizacja
├── agents/
│   ├── base_agent.py
│   ├── early_agent.py
│   └── mid_agent.py
├── models/              # Katalog na wytrenowane modele
│   ├── early.pt
│   └── mid.pt
├── wyniki/              # Katalog na wyniki treningów
│   ├── wyniki_early.csv
│   └── wyniki_mid.csv
├── training/            # Katalog na trenerów
│   ├──  train_early.py
│   └── train_mid.py
├── init__.py
└── play.py
```
