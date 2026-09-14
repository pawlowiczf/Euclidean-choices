### Trik: policz odpowiedź skupiska raz

Skupisko postawione w punkcie `mu` zawsze daje ten sam oczekiwany wektor punktów. Policz to raz metodą Monte Carlo dla każdego kandydata na środek skupiska. Potem mieszanka skupisk to już tylko **suma ważona wierszy tabelki**, bez żadnej symulacji.

Nierówność Hoeffdinga.

Kandydaci ustaleni. Wylosować położenia wyborców tak, żeby Plurality, Borda i Veto dały trzech różnych zwycięzców.

### Rozwiązanie

Trzy kroki, wszystko już napisane w poprzednich wiadomościach:

**1. Tabelka.** Siatka środków skupisk, kilka wartości `sigma`. Dla każdej pary policz raz metodą Monte Carlo, jakie punkty daje takie skupisko każdemu kandydatowi. To funkcja `blob_table`.

**2. Szukanie.** Losuj 3 do 5 wierszy z tabelki, losuj wagi z Dirichleta (`alpha` około 0.4, żeby wagi były skośne), zsumuj ważone wiersze, sprawdź zwycięzców. Trafienia zapisuj do słownika. To funkcja `search`. Milion prób idzie w sekundy, bo nic tu nie symulujesz.

**3. Weryfikacja.** Tabelka daje wartości oczekiwane, czyli granicę przy nieskończonej liczbie wyborców. Wylosuj z trafionej mieszanki realnych `N` wyborców, powtórz 200 razy i sprawdź, w ilu przypadkach zwycięzcy faktycznie są różni. To funkcja `verify`.

Krok 3 nie jest formalnością. Jeśli przewaga w tabelce była minimalna, przy realnym losowaniu zwycięzca potrafi się odwrócić i `verify` zwróci np. 0.4 zamiast 0.95.

### Co odpalić najpierw

Weź `m = 5` kandydatów, siatkę 12x12, `sigmas = [0.15, 0.3, 0.6]`, `alpha = 0.4`. Zobacz, ile trójek wpadło do słownika i jakie mają wyniki `verify`.

Wezmę maleńki konkretny przykład i policzę wszystko na prawdziwych liczbach.

**Układ:** 4 kandydatów. A=(0,1), B=(-1,-1), C=(1,-1), D=(0,0), czyli D w środku.

### 1. Czym jest siatka i jej gęstość

Siatka to po prostu **lista miejsc, w których wolno postawić skupisko wyborców**. Nic więcej. Bierzemy prostokąt i rozkładamy w nim punkty równomiernie.

Siatka 3x3 na obszarze od -2 do 2 to dokładnie te 9 punktów:

```
row 0: (-2,-2)   row 1: (0,-2)   row 2: (2,-2)
row 3: (-2, 0)   row 4: (0, 0)   row 5: (2, 0)
row 6: (-2, 2)   row 7: (0, 2)   row 8: (2, 2)
```

**Gęstość** to ile punktów na bok: 3x3 = 9 miejsc, 6x6 = 36 miejsc, 20x20 = 400 miejsc. **Zasięg** to jak daleko sięga prostokąt: ±2, ±5, ±8.

To są dwie niezależne rzeczy i właśnie dlatego mierzyłem je osobno. Zasięg okazał się ważny, gęstość prawie nieistotna.

### 2. Czym jest tabelka

Dla każdego z tych 9 miejsc stawiam skupisko (4000 wyborców rozsypanych wokół z `sigma = 0.3`) i liczę, ile punktów dostaje każdy kandydat **średnio od jednego wyborcy**. Oto prawdziwy wynik dla Plurality (kolumny to A, B, C, D):

```
row 0 (-2,-2):  [0.00  1.00  0.00  0.00]
row 3 (-2, 0):  [0.01  0.98  0.00  0.00]
row 4 ( 0, 0):  [0.05  0.01  0.01  0.93]
row 6 (-2, 2):  [1.00  0.00  0.00  0.00]
```

Czytaj wiersz 0 tak: skupisko w lewym dolnym rogu leży najbliżej B, więc **100% jego wyborców daje pierwsze miejsce kandydatowi B**. Wiersz 6, lewy górny róg, jest najbliżej A, więc 100% pierwszych miejsc idzie do A.

Ta sama tabelka istnieje osobno dla Bordy i Veto. Dla Bordy wiersz 0 wygląda tak:

```
row 0: [0.01  3.00  0.99  1.99]
```

Czyli wyborcy z lewego dolnego rogu dają B średnio 3 punkty (zawsze pierwszy), D około 2 (zawsze drugi), C około 1, A prawie 0 (zawsze ostatni).

Tabelkę liczysz **raz**.

### 3. Czym są wagi

Wagi mówią, **jaka część wyborców trafia do którego skupiska**. Muszą sumować się do 1, bo to udziały całości.

Przykład: wybieram wiersze 3, 6 i 0, i daję im wagi 0.21, 0.55, 0.23. Znaczy to:

- 21% wyborców stoi w punkcie (-2, 0)
- 55% wyborców stoi w punkcie (-2, 2)
- 23% wyborców stoi w punkcie (-2, -2)

Przy 1000 wyborcach to 210, 550 i 230 osób.

W kodzie: `w = rng.random(3); w /= w.sum()`. Losujesz trzy liczby i dzielisz przez sumę, żeby dały 1.

### 4. Czym jest wzór na wynik

Skoro wagi to udziały wyborców, a tabelka mówi, ile punktów daje jeden wyborca z danego miejsca, to wystarczy **pomnożyć i dodać**. Na prawdziwych liczbach:

```
0.21 * [0.01 0.98 0.00 0.00]   (wiersz 3)
0.55 * [1.00 0.00 0.00 0.00]   (wiersz 6)
0.23 * [0.00 1.00 0.00 0.00]   (wiersz 0)
------------------------------------------
       [0.56 0.44 0.00 0.00]
```

Największa liczba jest w kolumnie A, więc **A wygrywa Plurality**.

To samo robisz z tabelką Bordy i Veto na tych samych trzech wierszach i tych samych wagach:

```
Borda: [1.89  1.89  0.23  1.98]  ->  wygrywa D
Veto:  [0.77  1.00  0.23  1.00]  ->  wygrywa B
```

I masz to, czego szukasz: **Plurality daje A, Borda daje D, Veto daje B**. Trzech różnych zwycięzców.

Zapis `w @ P[idx]` w numpy to dokładnie ta operacja mnożenia i dodawania, tylko zapisana jednym znakiem.

### Dlaczego to jest szybkie

W pętli szukania nie symulujesz żadnych wyborców. Losujesz trzy numery wierszy i trzy wagi, wykonujesz powyższe mnożenie, patrzysz na `argmax`. To kilkanaście operacji arytmetycznych zamiast liczenia odległości dla tysiąca wyborców.

Dlatego setki tysięcy prób idą w kilka sekund, a cała symulacja została wykonana raz, przy budowie tabelki.