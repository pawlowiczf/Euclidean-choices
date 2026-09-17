Voting system:
$f: (C,V) \to 2^C$ 

Social choice function:
$f:(C,V) \to C$

Social welfare function:
$f:(C,V) \to p(C)$

Zwycięzcą Condorceta jest kandydat, który wygrywa wszystkie swoje pojedynki, czyli liczba wygranych to $m-1$, gdzie $m$ to liczba kandydatów.

#### Strict linear order
### McGarvey's construction

Dla każdej pary kandydatów $(x,y)$, którą rozstrzygamy na korzyść $x$, dodaj dwóch wyborców o pełnych rankingach (dowolnych), które są dokładnymi odwrotnościami, ale z jednym wyjątkiem: pozycja $x$ względem $y$ ma być taka sama u obu (czyli $x$ przed $y$ w obu rankingach).

### Condorcet ILP

Do znalezienia minimalnej liczby wyborców o danych preferencjach, których głosowanie ma konkretny rezultat można użyć ILP.
$$x_\sigma \in \mathbb{Z}_{\geq 0} \qquad \text{(number of voters with ranking } \sigma \text{)}$$
dla każdej permutacji $\sigma$ kandydatów — łącznie $m!$ zmiennych.

Po jednym na parę kandydatów $(i,j)$ - łącznie $\binom{m}{2}$ ograniczeń. 

Jeśli  **$i$ wygrywa z $j$** $$\sum_{\sigma:\ i \succ_\sigma j} x_\sigma \;-\; \sum_{\sigma:\ j \succ_\sigma i} x_\sigma \;\geq\; 1$$
Jeśli **remis między $i$ i $j$** $$\sum_{\sigma:\ i \succ_\sigma j} x_\sigma \;-\; \sum_{\sigma:\ j \succ_\sigma i} x_\sigma \;=\; 0$$ Funkcja celu to $$\min \sum_{\sigma} x_\sigma \qquad \text{(minimize total number of voters)}$$

W celu uzyskania dokładnej liczby potrzebnych rankingów możemy zmodyfikować model. Zamiast nierówności, każda para $(i,j)$ ma **dokładną równość**:

$$\sum_{\sigma:\ i \succ_\sigma j} x_\sigma - \sum_{\sigma:\ j \succ_\sigma i} x_\sigma = d_{ij}$$

gdzie $d_{ij}$ = zadana przewaga głosów (np. $C$ vs $B$: 5:1 → $d_{CB}=4$).

Dodatkowo, spójność z liczbą wyborców $n$:

$$\sum_{\sigma:\ i \succ_\sigma j} x_\sigma + \sum_{\sigma:\ j \succ_\sigma i} x_\sigma = n$$

### Twierdzenie Stearnsa (1959)

Dla dowolnego turnieju na $n$ kandydatach istnieje profil preferencji realizujący go przy użyciu co najwyżej: $$O\left(\frac{n}{\log n}\right)$$
