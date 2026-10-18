# Narrator model comparison (seed3001, hints mode, measurement only)

Same commentated moments, prompt, frames, phrase table and Verifier for every model; MAX_ATTEMPTS 3. Confound: gpt-6.1-sol runs at temperature 1 (it rejects any other value), with reasoning_effort=low; the gpt-4.1 models at temperature 0.2.

## Summary

| model | params | lines | first try | retried | rejected | calls (errors) | call p50/p95 s | line p50/p95 s | tokens/line in/out (reasoning) | cost/bundle USD |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | temperature=0.2, timeout_s=15.0 | 17 | 15 | 1 | 1 | 18 (1) | 1.6 / 8.1 | 1.6 / 15.0 | 3567 / 94 (0) | 0.0268 |
| gpt-6.1-sol | max_completion_tokens=2000, reasoning_effort=low, timeout_s=60.0 | 17 | 17 | 0 | 0 | 17 (0) | 4.6 / 17.2 | 4.6 / 17.3 | 3563 / 147 (45) | 0.1461 |

Line latency is the Narrator's wall time for a moment (all its attempts). Tokens per line are all calls' tokens over the commentated moments (retries included). Cost uses PRICES in the script (n/a: no price set).

## Repetition (scripts/line_variety.py)

| model | lang | n | near-dup | first6 | skel-dup | mean-sim | top opener |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | EN | 16 | 25% | 50% | 0% | 0.41 | 3/16 'the pass from' |
| gpt-4.1-mini | ES | 16 | 44% | 56% | 0% | 0.46 | 3/16 'un pase de' |
| gpt-4.1-mini | EL | 16 | 69% | 56% | 0% | 0.43 | 3/16 'η μπαλιά του' |
| gpt-6.1-sol | EN | 17 | 24% | 65% | 0% | 0.41 | 3/17 'even though it' |
| gpt-6.1-sol | ES | 17 | 71% | 59% | 0% | 0.48 | 3/17 '<P> da un' |
| gpt-6.1-sol | EL | 17 | 76% | 59% | 0% | 0.44 | 3/17 'ο <P> δίνει' |

## Verifier reasons (retried and rejected moments)

### gpt-4.1-mini
- [74'] Garus Dornic (retried)
  - attempt 1: length: line_en has 27 words, max 24
- [82'] Torian Kestley (rejected)
  - attempt 1: timeout after 15 s

### gpt-6.1-sol
- none

## English lines side by side

### [11'] Fenian Naskova | surprise_failure | very easy | frame reason
- **gpt-4.1-mini**: Fenian Naskova's pass was very easy because it came from their own defensive third, but it does not find a teammate.
- **gpt-6.1-sol**: Fenian Naskova's pass was very easy because it came from deep, but it does not find a teammate.

### [21'] Rosel Frossmont | surprise_success | very difficult | frame concession
- **gpt-4.1-mini**: The pass from Rosel Frossmont comes off, even though it was lofted high and played with a header.
- **gpt-6.1-sol**: The very difficult pass from Rosel Frossmont comes off, even though it was lofted high and played with a header.

### [31'] Garan Ravsen | surprise_failure | very easy | frame concession_first
- **gpt-4.1-mini**: Even though it came from their own defensive third, Garan Ravsen's very easy pass goes astray.
- **gpt-6.1-sol**: Even though it came from deep, Garan Ravsen's very easy pass goes astray.

### [31'] Loel Stavmont | surprise_failure | very easy | frame player_first
- **gpt-4.1-mini**: Loel Stavmont plays a very easy pass that does not reach a teammate, even though it came from their own defensive third.
- **gpt-6.1-sol**: Loel Stavmont plays a very easy pass that does not reach a teammate, even though it came from deep.

### [41'] Nevek Iversani | surprise_success | difficult | frame tier_first
- **gpt-4.1-mini**: Difficult because it was lofted high, yet Nevek Iversani's pass reaches a teammate.
- **gpt-6.1-sol**: It was difficult because it was lofted high, yet Nevek Iversani's pass reaches a teammate.

### [47'] Nevek Iversani | surprise_failure | easy | frame reason
- **gpt-4.1-mini**: The pass from Nevek Iversani was easy because it came from their own defensive third, but it is lost.
- **gpt-6.1-sol**: The pass from Nevek Iversani was easy because it came from deep, but it is lost.

### [51'] Yanir Iversmont | surprise_success | tricky | frame concession
- **gpt-4.1-mini**: Yanir Iversmont's tricky pass finds a teammate, even though it was lofted high.
- **gpt-6.1-sol**: Yanir Iversmont's tricky pass finds a teammate, even though it was lofted high.

### [57'] Jorian Wendwick | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini**: Even though it came from their own defensive third, Jorian Wendwick's easy pass does not reach a teammate.
- **gpt-6.1-sol**: Even though it came from deep, Jorian Wendwick's easy pass does not reach a teammate.

### [62'] Quinon Ostley | surprise_failure | easy | frame player_first
- **gpt-4.1-mini**: Quinon Ostley plays an easy pass that does not find a teammate, even though it was played along the ground.
- **gpt-6.1-sol**: Quinon Ostley plays an easy pass that does not find a teammate, even though it was played along the ground.

### [63'] Fenan Morrford | surprise_success | tricky | frame tier_first
- **gpt-4.1-mini**: Tricky because it was lofted high, yet Fenan Morrford's pass comes off.
- **gpt-6.1-sol**: It was tricky because it was lofted high, but Fenan Morrford's pass comes off.

### [67'] Nevek Iversani | surprise_success | tricky | frame reason
- **gpt-4.1-mini**: Nevek Iversani's pass was tricky because it was lofted high, but it finds a teammate.
- **gpt-6.1-sol**: Nevek Iversani's pass was tricky because it was lofted high, but it finds a teammate.

### [74'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini**: The pass from Garus Dornic is easy but does not reach a teammate, even though it was played along the ground.
- **gpt-6.1-sol**: The easy pass from Garus Dornic does not reach a teammate, even though it was played along the ground in midfield.

### [75'] Wesar Pellwick | surprise_success | tricky | frame concession_first
- **gpt-4.1-mini**: Even though it was lofted high, Wesar Pellwick's tricky pass reaches a teammate under pressure.
- **gpt-6.1-sol**: Even though it was lofted high and played under pressure, Wesar Pellwick's tricky pass reaches a teammate.

### [76'] Nevek Iversani | surprise_success | difficult | frame player_first
- **gpt-4.1-mini**: Nevek Iversani plays a difficult pass that comes off, even though it was lofted high and under pressure.
- **gpt-6.1-sol**: Nevek Iversani plays a difficult pass that comes off, even though it was lofted high and played under pressure.

### [80'] Jorian Wendwick | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini**: Easy because it was played along the ground, yet Jorian Wendwick's pass goes astray.
- **gpt-6.1-sol**: It was easy because it was played along the ground, but Jorian Wendwick's pass goes astray.

### [82'] Torian Kestley | surprise_failure | easy | frame reason
- **gpt-4.1-mini**: (no line: rejected)
- **gpt-6.1-sol**: The pass from Torian Kestley was easy because it came from deep, but does not reach a teammate.

### [88'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini**: Garus Dornic's easy pass does not find a teammate, even though it came from their own defensive third and was played along the ground.
- **gpt-6.1-sol**: Garus Dornic's easy pass does not find a teammate, even though it came from deep and went along the ground.

## Appendix: Spanish and Greek

### [11'] Fenian Naskova | surprise_failure | very easy | frame reason
- **gpt-4.1-mini ES**: El pase de Fenian Naskova era muy fácil porque salía de su propio tercio defensivo, pero no encuentra a un compañero.
- **gpt-6.1-sol ES**: El pase de Fenian Naskova era muy fácil porque venía desde atrás, pero no encuentra a un compañero.
- **gpt-4.1-mini EL**: Η πάσα του Fenian Naskova ήταν πολύ εύκολη λόγω του ότι έγινε από την άμυνα, αλλά δεν βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Η πάσα του Fenian Naskova ήταν πολύ εύκολη γιατί έγινε από την άμυνα, αλλά δεν βρίσκει συμπαίκτη.

### [21'] Rosel Frossmont | surprise_success | very difficult | frame concession
- **gpt-4.1-mini ES**: Un pase de Rosel Frossmont sale bien, aunque iba por alto y fue de cabeza.
- **gpt-6.1-sol ES**: Un pase muy difícil de Rosel Frossmont sale bien, aunque iba por alto y se dio de cabeza.
- **gpt-4.1-mini EL**: Η μπαλιά του Rosel Frossmont πετυχαίνει, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.
- **gpt-6.1-sol EL**: Η πολύ δύσκολη μπαλιά του Rosel Frossmont πετυχαίνει, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.

### [31'] Garan Ravsen | surprise_failure | very easy | frame concession_first
- **gpt-4.1-mini ES**: Aunque salió desde su propio tercio defensivo, el pase muy fácil de Garan Ravsen no llega a nada.
- **gpt-6.1-sol ES**: Aunque venía desde atrás, el pase muy fácil de Garan Ravsen no llega a nada.
- **gpt-4.1-mini EL**: Παρόλο που έγινε από την άμυνα, η πολύ εύκολη πάσα του Garan Ravsen δεν καταλήγει πουθενά.
- **gpt-6.1-sol EL**: Παρόλο που ήρθε από την άμυνα, η πολύ εύκολη πάσα του Garan Ravsen δεν καταλήγει πουθενά.

### [31'] Loel Stavmont | surprise_failure | very easy | frame player_first
- **gpt-4.1-mini ES**: Loel Stavmont juega un pase muy fácil que no llega a un compañero, aunque salió desde su propio tercio defensivo.
- **gpt-6.1-sol ES**: Loel Stavmont da un pase muy fácil que no llega a un compañero, aunque salía desde atrás.
- **gpt-4.1-mini EL**: Ο Loel Stavmont παίζει μια πολύ εύκολη πάσα που δεν φτάνει σε συμπαίκτη, παρόλο που έγινε από την άμυνα.
- **gpt-6.1-sol EL**: Ο Loel Stavmont δίνει μια πολύ εύκολη πάσα που δεν φτάνει σε συμπαίκτη, παρόλο που έγινε από την άμυνα.

### [41'] Nevek Iversani | surprise_success | difficult | frame tier_first
- **gpt-4.1-mini ES**: Difícil porque iba por alto, pero el pase de Nevek Iversani llega a un compañero.
- **gpt-6.1-sol ES**: Era difícil porque iba por alto, pero el pase de Nevek Iversani llega a un compañero.
- **gpt-4.1-mini EL**: Δύσκολη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Nevek Iversani φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Ήταν δύσκολη επειδή ήταν ψηλή, αλλά η πάσα του Nevek Iversani φτάνει σε συμπαίκτη.

### [47'] Nevek Iversani | surprise_failure | easy | frame reason
- **gpt-4.1-mini ES**: Un pase de Nevek Iversani era fácil porque salía desde su propio tercio defensivo, pero se pierde.
- **gpt-6.1-sol ES**: Un pase de Nevek Iversani era fácil porque venía desde atrás, pero se pierde.
- **gpt-4.1-mini EL**: Η μπαλιά του Nevek Iversani ήταν εύκολη λόγω του ότι έγινε από την άμυνα, αλλά χάνεται.
- **gpt-6.1-sol EL**: Η μπαλιά του Nevek Iversani ήταν εύκολη επειδή έγινε από την άμυνα, αλλά χάνεται.

### [51'] Yanir Iversmont | surprise_success | tricky | frame concession
- **gpt-4.1-mini ES**: El pase complicado de Yanir Iversmont encuentra a un compañero, aunque iba por alto.
- **gpt-6.1-sol ES**: El pase complicado de Yanir Iversmont encuentra a un compañero, aunque iba por alto.
- **gpt-4.1-mini EL**: Η περίπλοκη πάσα του Yanir Iversmont βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή.
- **gpt-6.1-sol EL**: Η περίπλοκη πάσα του Yanir Iversmont βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή.

### [57'] Jorian Wendwick | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini ES**: Aunque salió desde su propio tercio defensivo, el pase fácil de Jorian Wendwick no llega a un compañero.
- **gpt-6.1-sol ES**: Aunque salió desde atrás, el pase fácil de Jorian Wendwick no llega a un compañero.
- **gpt-4.1-mini EL**: Παρόλο που έγινε από την άμυνα, η εύκολη πάσα του Jorian Wendwick δεν φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Παρόλο που έγινε από την άμυνα, η εύκολη πάσα του Jorian Wendwick δεν φτάνει σε συμπαίκτη.

### [62'] Quinon Ostley | surprise_failure | easy | frame player_first
- **gpt-4.1-mini ES**: Quinon Ostley hace un pase fácil que no encuentra a un compañero, aunque iba raso.
- **gpt-6.1-sol ES**: Quinon Ostley da un pase fácil que no encuentra a un compañero, aunque iba raso.
- **gpt-4.1-mini EL**: Ο Quinon Ostley παίζει μια εύκολη πάσα που δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή.
- **gpt-6.1-sol EL**: Ο Quinon Ostley δίνει μια εύκολη πάσα που δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή.

### [63'] Fenan Morrford | surprise_success | tricky | frame tier_first
- **gpt-4.1-mini ES**: Complicado porque iba por alto, pero el pase de Fenan Morrford sale bien.
- **gpt-6.1-sol ES**: Era complicado porque iba por alto, pero el pase de Fenan Morrford sale bien.
- **gpt-4.1-mini EL**: Ήταν περίπλοκη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Fenan Morrford πετυχαίνει.
- **gpt-6.1-sol EL**: Ήταν περίπλοκη επειδή ήταν ψηλή, αλλά η πάσα του Fenan Morrford πετυχαίνει.

### [67'] Nevek Iversani | surprise_success | tricky | frame reason
- **gpt-4.1-mini ES**: El pase de Nevek Iversani era complicado porque iba por alto, pero encuentra a un compañero.
- **gpt-6.1-sol ES**: El pase de Nevek Iversani era complicado porque iba por alto, pero encuentra a un compañero.
- **gpt-4.1-mini EL**: Η πάσα του Nevek Iversani ήταν περίπλοκη λόγω του ότι ήταν ψηλή, αλλά βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Η πάσα του Nevek Iversani ήταν περίπλοκη επειδή ήταν ψηλή, αλλά βρίσκει συμπαίκτη.

### [74'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini ES**: Un pase de Garus Dornic es fácil pero no llega a un compañero, aunque iba raso.
- **gpt-6.1-sol ES**: Un pase fácil de Garus Dornic no llega a un compañero, aunque iba raso en el centro del campo.
- **gpt-4.1-mini EL**: Η μπαλιά του Garus Dornic ήταν εύκολη αλλά δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή.
- **gpt-6.1-sol EL**: Η εύκολη μπαλιά του Garus Dornic δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή από το κέντρο.

### [75'] Wesar Pellwick | surprise_success | tricky | frame concession_first
- **gpt-4.1-mini ES**: Aunque iba por alto, el pase complicado de Wesar Pellwick llega a un compañero bajo presión.
- **gpt-6.1-sol ES**: Aunque iba por alto y bajo presión, el pase complicado de Wesar Pellwick llega a un compañero.
- **gpt-4.1-mini EL**: Παρόλο που ήταν ψηλή, η περίπλοκη πάσα του Wesar Pellwick φτάνει σε συμπαίκτη υπό πίεση.
- **gpt-6.1-sol EL**: Παρόλο που ήταν ψηλή και έγινε υπό πίεση, η περίπλοκη πάσα του Wesar Pellwick φτάνει σε συμπαίκτη.

### [76'] Nevek Iversani | surprise_success | difficult | frame player_first
- **gpt-4.1-mini ES**: Nevek Iversani juega un pase difícil que sale bien, aunque iba por alto y bajo presión.
- **gpt-6.1-sol ES**: Nevek Iversani da un pase difícil que sale bien, aunque iba por alto y bajo presión.
- **gpt-4.1-mini EL**: Ο Nevek Iversani παίζει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και υπό πίεση.
- **gpt-6.1-sol EL**: Ο Nevek Iversani δίνει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και έγινε υπό πίεση.

### [80'] Jorian Wendwick | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini ES**: Fácil porque iba raso, pero el pase de Jorian Wendwick no llega a nada.
- **gpt-6.1-sol ES**: Era fácil porque iba raso, pero el pase de Jorian Wendwick no llega a nada.
- **gpt-4.1-mini EL**: Εύκολη διότι ήταν συρτή, αλλά η πάσα του Jorian Wendwick δεν καταλήγει πουθενά.
- **gpt-6.1-sol EL**: Ήταν εύκολη επειδή ήταν συρτή, αλλά η πάσα του Jorian Wendwick δεν καταλήγει πουθενά.

### [82'] Torian Kestley | surprise_failure | easy | frame reason
- **gpt-4.1-mini ES**: (no line: rejected)
- **gpt-6.1-sol ES**: Un pase de Torian Kestley era fácil porque venía desde atrás, pero no llega a un compañero.
- **gpt-4.1-mini EL**: (no line: rejected)
- **gpt-6.1-sol EL**: Η μπαλιά του Torian Kestley ήταν εύκολη επειδή έγινε από την άμυνα, αλλά δεν φτάνει σε συμπαίκτη.

### [88'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini ES**: El pase fácil de Garus Dornic no encuentra a un compañero, aunque salió desde su propio tercio defensivo y fue raso.
- **gpt-6.1-sol ES**: El pase fácil de Garus Dornic no encuentra a un compañero, aunque venía desde atrás e iba raso.
- **gpt-4.1-mini EL**: Η εύκολη πάσα του Garus Dornic δεν βρίσκει συμπαίκτη, παρόλο που έγινε από την άμυνα και ήταν συρτή.
- **gpt-6.1-sol EL**: Η εύκολη πάσα του Garus Dornic δεν βρίσκει συμπαίκτη, παρόλο που έγινε από την άμυνα και ήταν συρτή.
