# Narrator model comparison (seed3001, hints mode, measurement only)

Same commentated moments, prompt, frames, phrase table and Verifier for every model; MAX_ATTEMPTS 3. Confound: gpt-6.1-sol runs at temperature 1 (it rejects any other value), with reasoning_effort=low; the gpt-4.1 models at temperature 0.2.

## Summary

| model | params | lines | first try | retried | rejected | calls (errors) | call p50/p95 s | line p50/p95 s | tokens/line in/out (reasoning) | cost/bundle USD |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | temperature=0.2, timeout_s=15.0 | 17 | 17 | 0 | 0 | 17 (0) | 1.4 / 8.1 | 1.4 / 8.1 | 3517 / 93 (0) | 0.0265 |
| gpt-4.1 | temperature=0.2, timeout_s=15.0 | 17 | 17 | 0 | 0 | 17 (0) | 1.3 / 8.2 | 1.3 / 8.2 | 3517 / 93 (0) | 0.1322 |
| gpt-6.1-sol | max_completion_tokens=2000, reasoning_effort=low, timeout_s=60.0 | 17 | 17 | 0 | 0 | 17 (0) | 4.4 / 6.7 | 4.4 / 6.7 | 3514 / 150 (48) | n/a |

Line latency is the Narrator's wall time for a moment (all its attempts). Tokens per line are all calls' tokens over the commentated moments (retries included). Cost uses PRICES in the script (n/a: no price set).

## Repetition (scripts/line_variety.py)

| model | lang | n | near-dup | first6 | skel-dup | mean-sim | top opener |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | EN | 17 | 24% | 59% | 0% | 0.41 | 3/17 '<P> s pass' |
| gpt-4.1-mini | ES | 17 | 65% | 76% | 0% | 0.47 | 4/17 'un pase de' |
| gpt-4.1-mini | EL | 17 | 65% | 65% | 0% | 0.43 | 4/17 'η μπαλιά του' |
| gpt-4.1 | EN | 17 | 24% | 47% | 0% | 0.42 | 4/17 'the pass from' |
| gpt-4.1 | ES | 17 | 76% | 59% | 0% | 0.48 | 4/17 'un pase de' |
| gpt-4.1 | EL | 17 | 82% | 59% | 0% | 0.44 | 4/17 'η μπαλιά του' |
| gpt-6.1-sol | EN | 17 | 35% | 65% | 0% | 0.42 | 3/17 'even though it' |
| gpt-6.1-sol | ES | 17 | 71% | 59% | 0% | 0.48 | 3/17 '<P> da un' |
| gpt-6.1-sol | EL | 17 | 76% | 59% | 0% | 0.45 | 3/17 '<P> δίνει μια' |

## Verifier reasons (retried and rejected moments)

### gpt-4.1-mini
- none

### gpt-4.1
- none

### gpt-6.1-sol
- none

## English lines side by side

### [11'] Fenian Naskova | surprise_failure | very easy | frame reason
- **gpt-4.1-mini**: Fenian Naskova's pass was very easy because it came from their own defensive third, but it does not find a teammate.
- **gpt-4.1**: Fenian Naskova's pass was very easy because it came from their own defensive third, but it does not find a teammate.
- **gpt-6.1-sol**: Fenian Naskova's pass was very easy because it came from deep, but it does not find a teammate.

### [21'] Rosel Frossmont | surprise_success | very difficult | frame concession
- **gpt-4.1-mini**: Rosel Frossmont's pass comes off, even though it was lofted high and played with a header.
- **gpt-4.1**: The pass from Rosel Frossmont comes off, even though it was lofted high and played with a header.
- **gpt-6.1-sol**: The very difficult pass from Rosel Frossmont comes off, even though it was lofted high and played with a header.

### [31'] Garan Ravsen | surprise_failure | very easy | frame concession_first
- **gpt-4.1-mini**: Even though it came from their own defensive third, Garan Ravsen's very easy pass goes astray.
- **gpt-4.1**: Even though it was played along the ground, Garan Ravsen's very easy pass goes astray.
- **gpt-6.1-sol**: Even though it came from deep, Garan Ravsen's very easy pass goes astray.

### [31'] Loel Stavmont | surprise_failure | very easy | frame player_first
- **gpt-4.1-mini**: Loel Stavmont plays a very easy pass that does not reach a teammate, even though it came from their own defensive third.
- **gpt-4.1**: Loel Stavmont plays a very easy pass that does not reach a teammate, even though it was played along the ground.
- **gpt-6.1-sol**: Loel Stavmont plays a very easy pass that does not reach a teammate, even though it came from deep.

### [41'] Nevek Iversani | surprise_success | difficult | frame tier_first
- **gpt-4.1-mini**: Difficult because it was lofted high, yet Nevek Iversani's pass reaches a teammate.
- **gpt-4.1**: Difficult because it was lofted high, yet Nevek Iversani's pass reaches a teammate.
- **gpt-6.1-sol**: It was difficult because it was lofted high, yet Nevek Iversani's pass reaches a teammate.

### [47'] Nevek Iversani | surprise_failure | easy | frame reason
- **gpt-4.1-mini**: The pass from Nevek Iversani was easy because it came from their own defensive third, but it is lost.
- **gpt-4.1**: The pass from Nevek Iversani was easy because it came from their own defensive third, but it is lost.
- **gpt-6.1-sol**: The pass from Nevek Iversani was easy because it came from deep, but it is lost.

### [51'] Yanir Iversmont | surprise_success | tricky | frame concession
- **gpt-4.1-mini**: Yanir Iversmont's tricky pass finds a teammate, even though it was lofted high.
- **gpt-4.1**: Yanir Iversmont's tricky pass finds a teammate, even though it was lofted high.
- **gpt-6.1-sol**: Yanir Iversmont's tricky pass finds a teammate, even though it was lofted high.

### [57'] Jorian Wendwick | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini**: Even though it came from their own defensive third, Jorian Wendwick's easy pass does not reach a teammate.
- **gpt-4.1**: Even though it came from their own defensive third, Jorian Wendwick's easy pass does not reach a teammate.
- **gpt-6.1-sol**: Even though it came from deep, Jorian Wendwick's easy pass does not reach a teammate.

### [62'] Quinon Ostley | surprise_failure | easy | frame player_first
- **gpt-4.1-mini**: Quinon Ostley plays an easy pass that does not find a teammate, even though it was along the ground.
- **gpt-4.1**: Quinon Ostley plays an easy pass that does not find a teammate, even though it was played along the ground.
- **gpt-6.1-sol**: Quinon Ostley plays an easy pass that does not find a teammate, even though it was played along the ground.

### [63'] Fenan Morrford | surprise_success | tricky | frame tier_first
- **gpt-4.1-mini**: Tricky because it was lofted high, yet Fenan Morrford's pass comes off.
- **gpt-4.1**: Tricky because it was lofted high, yet Fenan Morrford's pass comes off.
- **gpt-6.1-sol**: It was tricky because it was lofted high, yet Fenan Morrford's pass comes off.

### [67'] Nevek Iversani | surprise_success | tricky | frame reason
- **gpt-4.1-mini**: Nevek Iversani's pass was tricky because it was lofted high, but it finds a teammate.
- **gpt-4.1**: Nevek Iversani's pass was tricky because it was lofted high, but it finds a teammate.
- **gpt-6.1-sol**: Nevek Iversani's pass was tricky because it was lofted high, but it finds a teammate.

### [74'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini**: The pass from Garus Dornic is easy but does not reach a teammate, even though it was played along the ground.
- **gpt-4.1**: The pass from Garus Dornic does not reach a teammate, even though it was played along the ground and in the middle third.
- **gpt-6.1-sol**: The easy pass from Garus Dornic does not reach a teammate, even though it was along the ground in midfield.

### [75'] Wesar Pellwick | surprise_success | tricky | frame concession_first
- **gpt-4.1-mini**: Even though it was lofted high and under pressure, Wesar Pellwick's tricky pass reaches a teammate.
- **gpt-4.1**: Even though it was lofted high and under pressure, Wesar Pellwick's tricky pass reaches a teammate.
- **gpt-6.1-sol**: Even though it was lofted high and played under pressure, Wesar Pellwick's tricky pass reaches a teammate.

### [76'] Nevek Iversani | surprise_success | difficult | frame player_first
- **gpt-4.1-mini**: Nevek Iversani plays a difficult pass that comes off, even though it was lofted high and under pressure.
- **gpt-4.1**: Nevek Iversani plays a difficult pass that comes off, even though it was lofted high and under pressure.
- **gpt-6.1-sol**: Nevek Iversani plays a difficult pass that comes off, even though it was lofted high and played under pressure.

### [80'] Jorian Wendwick | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini**: Easy because it was played along the ground, yet Jorian Wendwick's pass goes astray.
- **gpt-4.1**: Easy because it was played along the ground, yet Jorian Wendwick's pass goes astray.
- **gpt-6.1-sol**: It was easy because it was played along the ground, but Jorian Wendwick's pass goes astray.

### [82'] Torian Kestley | surprise_failure | easy | frame reason
- **gpt-4.1-mini**: The pass from Torian Kestley was easy because it came from their own defensive third, but it does not reach a teammate.
- **gpt-4.1**: The pass from Torian Kestley was easy because it came from their own defensive third, but it does not reach a teammate.
- **gpt-6.1-sol**: The pass from Torian Kestley was easy because it came from deep, but does not reach a teammate.

### [88'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini**: Garus Dornic's easy pass does not find a teammate, even though it came from their own defensive third and was played along the ground.
- **gpt-4.1**: Garus Dornic's pass does not find a teammate, even though it was played along the ground and from their own defensive third.
- **gpt-6.1-sol**: Garus Dornic's easy pass does not find a teammate, even though it was played along the ground from deep.

## Appendix: Spanish and Greek

### [11'] Fenian Naskova | surprise_failure | very easy | frame reason
- **gpt-4.1-mini ES**: El pase de Fenian Naskova era muy fácil porque salía de su propio tercio defensivo, pero no encuentra a un compañero.
- **gpt-4.1 ES**: El pase de Fenian Naskova era muy fácil porque salía desde su propio tercio defensivo, pero no encuentra a un compañero.
- **gpt-6.1-sol ES**: El pase de Fenian Naskova era muy fácil porque venía desde atrás, pero no encuentra a un compañero.
- **gpt-4.1-mini EL**: Η πάσα του Fenian Naskova ήταν πολύ εύκολη διότι έγινε από την άμυνα, αλλά δεν βρίσκει συμπαίκτη.
- **gpt-4.1 EL**: Η πάσα του Fenian Naskova ήταν πολύ εύκολη διότι έγινε από την άμυνα, αλλά δεν βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Η πάσα του Fenian Naskova ήταν πολύ εύκολη επειδή έγινε από την άμυνα, αλλά δεν βρίσκει συμπαίκτη.

### [21'] Rosel Frossmont | surprise_success | very difficult | frame concession
- **gpt-4.1-mini ES**: Un pase de Rosel Frossmont sale bien, aunque iba por alto y fue de cabeza.
- **gpt-4.1 ES**: Un pase de Rosel Frossmont sale bien, aunque iba por alto y fue de cabeza.
- **gpt-6.1-sol ES**: Un pase muy difícil de Rosel Frossmont sale bien, aunque iba por alto y era de cabeza.
- **gpt-4.1-mini EL**: Η μπαλιά του Rosel Frossmont πετυχαίνει, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.
- **gpt-4.1 EL**: Η μπαλιά του Rosel Frossmont πετυχαίνει, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.
- **gpt-6.1-sol EL**: Η πολύ δύσκολη μπαλιά του Rosel Frossmont πετυχαίνει, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.

### [31'] Garan Ravsen | surprise_failure | very easy | frame concession_first
- **gpt-4.1-mini ES**: Aunque salió desde su propio tercio defensivo, el pase muy fácil de Garan Ravsen no llega a nada.
- **gpt-4.1 ES**: Aunque iba raso, el pase muy fácil de Garan Ravsen no llega a nada.
- **gpt-6.1-sol ES**: Aunque venía desde atrás, el pase muy fácil de Garan Ravsen no llega a nada.
- **gpt-4.1-mini EL**: Παρόλο που έγινε από την άμυνα, η πολύ εύκολη πάσα του Garan Ravsen δεν καταλήγει πουθενά.
- **gpt-4.1 EL**: Παρόλο που ήταν συρτή, η πολύ εύκολη πάσα του Garan Ravsen δεν καταλήγει πουθενά.
- **gpt-6.1-sol EL**: Παρόλο που ήρθε από την άμυνα, η πολύ εύκολη πάσα του Garan Ravsen δεν καταλήγει πουθενά.

### [31'] Loel Stavmont | surprise_failure | very easy | frame player_first
- **gpt-4.1-mini ES**: Loel Stavmont juega un pase muy fácil que no llega a un compañero, aunque salió desde su propio tercio defensivo.
- **gpt-4.1 ES**: Loel Stavmont juega un pase muy fácil que no llega a un compañero, aunque iba raso.
- **gpt-6.1-sol ES**: Loel Stavmont da un pase muy fácil que no llega a un compañero, aunque salió desde atrás.
- **gpt-4.1-mini EL**: Ο Loel Stavmont παίζει μια πολύ εύκολη πάσα που δεν φτάνει σε συμπαίκτη, παρόλο που έγινε από την άμυνα.
- **gpt-4.1 EL**: Ο Loel Stavmont παίζει μια πολύ εύκολη πάσα που δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή.
- **gpt-6.1-sol EL**: Loel Stavmont δίνει μια πολύ εύκολη πάσα που δεν φτάνει σε συμπαίκτη, παρόλο που έγινε από την άμυνα.

### [41'] Nevek Iversani | surprise_success | difficult | frame tier_first
- **gpt-4.1-mini ES**: Difícil porque iba por alto, pero el pase de Nevek Iversani llega a un compañero.
- **gpt-4.1 ES**: Difícil porque iba por alto, pero el pase de Nevek Iversani llega a un compañero.
- **gpt-6.1-sol ES**: Era difícil porque iba por alto, pero el pase de Nevek Iversani llega a un compañero.
- **gpt-4.1-mini EL**: Δύσκολη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Nevek Iversani φτάνει σε συμπαίκτη.
- **gpt-4.1 EL**: Δύσκολη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Nevek Iversani φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Ήταν δύσκολη επειδή ήταν ψηλή, αλλά η πάσα του Nevek Iversani φτάνει σε συμπαίκτη.

### [47'] Nevek Iversani | surprise_failure | easy | frame reason
- **gpt-4.1-mini ES**: Un pase de Nevek Iversani era fácil porque salía desde su propio tercio defensivo, pero se pierde.
- **gpt-4.1 ES**: Un pase de Nevek Iversani era fácil porque salía desde su propio tercio defensivo, pero se pierde.
- **gpt-6.1-sol ES**: Un pase de Nevek Iversani era fácil porque venía desde atrás, pero se pierde.
- **gpt-4.1-mini EL**: Η μπαλιά του Nevek Iversani ήταν εύκολη λόγω του ότι έγινε από την άμυνα, αλλά χάνεται.
- **gpt-4.1 EL**: Η μπαλιά του Nevek Iversani ήταν εύκολη διότι έγινε από την άμυνα, αλλά χάνεται.
- **gpt-6.1-sol EL**: Η μπαλιά του Nevek Iversani ήταν εύκολη επειδή έγινε από την άμυνα, αλλά χάνεται.

### [51'] Yanir Iversmont | surprise_success | tricky | frame concession
- **gpt-4.1-mini ES**: El pase complicado de Yanir Iversmont encuentra a un compañero, aunque iba por alto.
- **gpt-4.1 ES**: El pase complicado de Yanir Iversmont encuentra a un compañero, aunque iba por alto.
- **gpt-6.1-sol ES**: El pase complicado de Yanir Iversmont encuentra a un compañero, aunque iba por alto.
- **gpt-4.1-mini EL**: Η περίπλοκη πάσα του Yanir Iversmont βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή.
- **gpt-4.1 EL**: Η περίπλοκη πάσα του Yanir Iversmont βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή.
- **gpt-6.1-sol EL**: Η περίπλοκη πάσα του Yanir Iversmont βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή.

### [57'] Jorian Wendwick | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini ES**: Aunque salió desde su propio tercio defensivo, el pase fácil de Jorian Wendwick no llega a un compañero.
- **gpt-4.1 ES**: Aunque salió desde su propio tercio defensivo, el pase fácil de Jorian Wendwick no llega a un compañero.
- **gpt-6.1-sol ES**: Aunque salía desde atrás, el pase fácil de Jorian Wendwick no llega a un compañero.
- **gpt-4.1-mini EL**: Παρόλο που έγινε από την άμυνα, η εύκολη πάσα του Jorian Wendwick δεν φτάνει σε συμπαίκτη.
- **gpt-4.1 EL**: Παρόλο που έγινε από την άμυνα, η εύκολη πάσα του Jorian Wendwick δεν φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Παρόλο που έγινε από την άμυνα, η εύκολη πάσα του Jorian Wendwick δεν φτάνει σε συμπαίκτη.

### [62'] Quinon Ostley | surprise_failure | easy | frame player_first
- **gpt-4.1-mini ES**: Quinon Ostley juega un pase fácil que no encuentra a un compañero, aunque iba raso.
- **gpt-4.1 ES**: Quinon Ostley juega un pase fácil que no encuentra a un compañero, aunque iba raso.
- **gpt-6.1-sol ES**: Quinon Ostley da un pase fácil que no encuentra a un compañero, aunque iba raso.
- **gpt-4.1-mini EL**: Ο Quinon Ostley παίζει μια εύκολη πάσα που δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή.
- **gpt-4.1 EL**: Ο Quinon Ostley παίζει μια εύκολη πάσα που δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή.
- **gpt-6.1-sol EL**: Quinon Ostley δίνει μια εύκολη πάσα που δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή.

### [63'] Fenan Morrford | surprise_success | tricky | frame tier_first
- **gpt-4.1-mini ES**: Complicado porque iba por alto, pero el pase de Fenan Morrford sale bien.
- **gpt-4.1 ES**: Complicado porque iba por alto, pero el pase de Fenan Morrford sale bien.
- **gpt-6.1-sol ES**: Era complicado porque iba por alto, pero el pase de Fenan Morrford sale bien.
- **gpt-4.1-mini EL**: Η πάσα του Fenan Morrford ήταν περίπλοκη λόγω του ότι ήταν ψηλή, αλλά πετυχαίνει.
- **gpt-4.1 EL**: Περίπλοκη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Fenan Morrford πετυχαίνει.
- **gpt-6.1-sol EL**: Ήταν περίπλοκη επειδή ήταν ψηλή, αλλά η πάσα του Fenan Morrford πετυχαίνει.

### [67'] Nevek Iversani | surprise_success | tricky | frame reason
- **gpt-4.1-mini ES**: El pase de Nevek Iversani era complicado porque iba por alto, pero encuentra a un compañero.
- **gpt-4.1 ES**: El pase de Nevek Iversani era complicado porque iba por alto, pero encuentra a un compañero.
- **gpt-6.1-sol ES**: El pase de Nevek Iversani era complicado porque iba por alto, pero encuentra a un compañero.
- **gpt-4.1-mini EL**: Η πάσα του Nevek Iversani ήταν περίπλοκη λόγω του ότι ήταν ψηλή, αλλά βρίσκει συμπαίκτη.
- **gpt-4.1 EL**: Η πάσα του Nevek Iversani ήταν περίπλοκη λόγω του ότι ήταν ψηλή, αλλά βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Η πάσα του Nevek Iversani ήταν περίπλοκη επειδή ήταν ψηλή, αλλά βρίσκει συμπαίκτη.

### [74'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini ES**: Un pase de Garus Dornic era fácil pero no llega a un compañero, aunque iba raso.
- **gpt-4.1 ES**: Un pase de Garus Dornic no llega a un compañero, aunque iba raso y en el tercio medio.
- **gpt-6.1-sol ES**: Un pase fácil de Garus Dornic no llega a un compañero, aunque iba raso en el tercio medio.
- **gpt-4.1-mini EL**: Η μπαλιά του Garus Dornic ήταν εύκολη αλλά δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή.
- **gpt-4.1 EL**: Η μπαλιά του Garus Dornic δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή και έγινε στο κέντρο.
- **gpt-6.1-sol EL**: Η εύκολη μπαλιά του Garus Dornic δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή στο κέντρο.

### [75'] Wesar Pellwick | surprise_success | tricky | frame concession_first
- **gpt-4.1-mini ES**: Aunque iba por alto y bajo presión, el pase complicado de Wesar Pellwick llega a un compañero.
- **gpt-4.1 ES**: Aunque iba por alto y bajo presión, el pase complicado de Wesar Pellwick llega a un compañero.
- **gpt-6.1-sol ES**: Aunque iba por alto y bajo presión, el complicado pase de Wesar Pellwick llega a un compañero.
- **gpt-4.1-mini EL**: Παρόλο που ήταν ψηλή και υπό πίεση, η περίπλοκη πάσα του Wesar Pellwick φτάνει σε συμπαίκτη.
- **gpt-4.1 EL**: Παρόλο που ήταν ψηλή και υπό πίεση, η περίπλοκη πάσα του Wesar Pellwick φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Παρόλο που ήταν ψηλή και έγινε υπό πίεση, η περίπλοκη πάσα του Wesar Pellwick φτάνει σε συμπαίκτη.

### [76'] Nevek Iversani | surprise_success | difficult | frame player_first
- **gpt-4.1-mini ES**: Nevek Iversani juega un pase difícil que sale bien, aunque iba por alto y bajo presión.
- **gpt-4.1 ES**: Nevek Iversani juega un pase difícil que sale bien, aunque iba por alto y bajo presión.
- **gpt-6.1-sol ES**: Nevek Iversani da un pase difícil que sale bien, aunque iba por alto y bajo presión.
- **gpt-4.1-mini EL**: Ο Nevek Iversani παίζει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και υπό πίεση.
- **gpt-4.1 EL**: Ο Nevek Iversani παίζει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και έγινε υπό πίεση.
- **gpt-6.1-sol EL**: Nevek Iversani δίνει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και έγινε υπό πίεση.

### [80'] Jorian Wendwick | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini ES**: Fácil porque iba raso, pero el pase de Jorian Wendwick no llega a nada.
- **gpt-4.1 ES**: Fácil porque iba raso, pero el pase de Jorian Wendwick no llega a nada.
- **gpt-6.1-sol ES**: Era fácil porque iba raso, pero el pase de Jorian Wendwick no llega a nada.
- **gpt-4.1-mini EL**: Εύκολη διότι ήταν συρτή, αλλά η πάσα του Jorian Wendwick δεν καταλήγει πουθενά.
- **gpt-4.1 EL**: Εύκολη λόγω του ότι ήταν συρτή, αλλά η πάσα του Jorian Wendwick δεν καταλήγει πουθενά.
- **gpt-6.1-sol EL**: Ήταν εύκολη επειδή ήταν συρτή, αλλά η πάσα του Jorian Wendwick δεν καταλήγει πουθενά.

### [82'] Torian Kestley | surprise_failure | easy | frame reason
- **gpt-4.1-mini ES**: Un pase de Torian Kestley era fácil porque salía desde su propio tercio defensivo, pero no llega a un compañero.
- **gpt-4.1 ES**: Un pase de Torian Kestley era fácil porque salía desde su propio tercio defensivo, pero no llega a un compañero.
- **gpt-6.1-sol ES**: Un pase de Torian Kestley era fácil porque venía desde atrás, pero no llega a un compañero.
- **gpt-4.1-mini EL**: Η μπαλιά του Torian Kestley ήταν εύκολη διότι έγινε από την άμυνα, αλλά δεν φτάνει σε συμπαίκτη.
- **gpt-4.1 EL**: Η μπαλιά του Torian Kestley ήταν εύκολη διότι έγινε από την άμυνα, αλλά δεν φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Η μπαλιά του Torian Kestley ήταν εύκολη επειδή έγινε από την άμυνα, αλλά δεν φτάνει σε συμπαίκτη.

### [88'] Garus Dornic | surprise_failure | easy | frame concession
- **gpt-4.1-mini ES**: El pase fácil de Garus Dornic no encuentra a un compañero, aunque salió desde su propio tercio defensivo y fue raso.
- **gpt-4.1 ES**: El pase de Garus Dornic no encuentra a un compañero, aunque iba raso y desde su propio tercio defensivo.
- **gpt-6.1-sol ES**: El pase fácil de Garus Dornic no encuentra a un compañero, aunque iba raso desde atrás.
- **gpt-4.1-mini EL**: Η εύκολη πάσα του Garus Dornic δεν βρίσκει συμπαίκτη, παρόλο που έγινε από την άμυνα και ήταν συρτή.
- **gpt-4.1 EL**: Η πάσα του Garus Dornic δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή και έγινε από την άμυνα.
- **gpt-6.1-sol EL**: Η εύκολη πάσα του Garus Dornic δεν βρίσκει συμπαίκτη, παρόλο που ήταν συρτή και έγινε από την άμυνα.
