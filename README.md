# Lab 04 - Projekt podsumowujący biblioteki Pandas i Matplotlib

Celem projektu jest utrwalenie wiedzy dotyczącej pobierania i wczytywania danych oraz biblioteki Pandas. W celu realizacji części zadań zapoznaj się z przykłądami i informacjami dotyczącymi przetwarzania danych tabelarycznych w tym mechanizmu wieloindeksowania.

Na stronie Social Security Administration umieszczono dane zawierającą listę imion oraz częstotliwość ich występowania w latach 1880-2019.

1. Wczytaj dane ze wszystkich plików do pojedynczej tablicy (używając Pandas).
2. Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie.
3. Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie rozróżniając imiona męskie i żeńskie.
4. Stwórz nowe kolumny frequency_male i frequency_female i określ popularność każdego z imion w danym każdym roku dzieląc liczbę razy, kiedy imię zostało nadane przez całkowita liczbę urodzeń dla danej płci.
5. Określ i wyświetl wykres złożony z dwóch podwykresów, gdzie osią x jest skala czasu, a oś y reprezentuje: 
  - liczbę urodzin w danym roku (wykres na górze)
  - stosunek liczby narodzin dziewczynek do liczby narodzin chłopców (wykres na dole) W którym roku zanotowano najmniejszą, a w którym największą różnicę w liczbie urodzeń między chłopcami a dziewczynkami (pytanie dotyczy podwykresu przedstawiającego stosunek liczby urodzin)?
6. Wyznacz 1000 najpopularniejszych imion dla każdej płci w całym zakresie czasowym, metoda powinna polegać na wyznaczeniu 1000 najpopularniejszych imion dla każdego roku i dla każdej płci a następnie ich zsumowaniu w celu ustalenia rankingu top 1000 dla każdej płci.
7. Wyświetl wykresy zmian dla imienia męskiego John pierwszego imienia w żeńskiego rankingu top-1000: 
  - na osi Y po lewej liczbę razy kiedy imę zostało nadane w każdym roku (zanotuj ile razy nadano to imię w 1930, 1970 i 2015r)?
  - na osi Y po prawej popularność tych imion w każdym z tych lat
8. Wykreśl wykres z podziałem na lata i płeć zawierający informację jaki procent w danym roku stanowiły imiona należące do rankingu top1000 (wyznaczonego dla całego zbioru). Wykres ten opisuje różnorodność imion, zanotuj rok w którym zaobserwowano największą różnicę w różnorodności między imionami męskimi a żeńskimi.
9. Zweryfikuj hipotezę czy prawdą jest, że w obserwowanym okresie rozkład ostatnich liter imion męskich uległ istotnej zmianie? W tym celu:
  - dokonaj agregacji wszystkich urodzeń w pełnym zbiorze danych z podziałem na rok i płeć i ostatnią literę,
  - wyodrębnij dane dla lat 1915, 1965, 2018
  - znormalizuj dane względem całkowitej liczby urodzin w danym roku
  - wyświetl dane popularności litery dla mężczyzn w postaci wykresu słupkowego zawierającego poszczególne lata i gdzie słupki grupowane są wg litery. Zanotuj, dla której litery wystąpił największy wzrost/spadek między rokiem 1915 a 2018)
  - Dla 3 liter dla których zaobserwowano największą zmianę wyświetl przebieg trendu popularności w czasie maksymalnym przedziale czasu
10. Znajdź imiona, które nadawane były zarówno dziewczynkom jak i chłopcom (zanotuj te które w całym horyzoncie czasu mają zbliżoną liczebność (stosunek nadanych imion meskich i żeńskich). Wyznacz imię, dla którego zaobserwowano "nieznaczną przewagę imion męskich)
12. Wczytaj dane z bazy opisującej śmiertelność w okresie od 1959-2018r w poszczególnych grupach wiekowych: USA_ltper_1x1.sqlite, opis: https://www.mortality.org/Public/ExplanatoryNotes.php. Spróbuj zagregować dane już na etapie zapytania SQL.
13. Wyznacz przyrost naturalny w analizowanym okresie
14. Wyznacz i wyświetl współczynnik przeżywalności dzieci w pierwszym roku życia w analizowanym okresie.
