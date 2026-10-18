# Narrator model comparison (seed42, hints mode, measurement only)

Same commentated moments, prompt, frames, phrase table and Verifier for every model; MAX_ATTEMPTS 3. Confound: gpt-6.1-sol runs at temperature 1 (it rejects any other value), with reasoning_effort=low; the gpt-4.1 models at temperature 0.2.

## Summary

| model | params | lines | first try | retried | rejected | calls (errors) | call p50/p95 s | line p50/p95 s | tokens/line in/out (reasoning) | cost/bundle USD |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | temperature=0.2, timeout_s=15.0 | 17 | 16 | 1 | 0 | 18 (0) | 1.8 / 10.7 | 1.9 / 10.7 | 3766 / 97 (0) | 0.0283 |
| gpt-6.1-sol | max_completion_tokens=2000, reasoning_effort=low, timeout_s=60.0 | 17 | 17 | 0 | 0 | 17 (0) | 3.0 / 7.8 | 3.0 / 7.8 | 3552 / 154 (52) | 0.1470 |

Line latency is the Narrator's wall time for a moment (all its attempts). Tokens per line are all calls' tokens over the commentated moments (retries included). Cost uses PRICES in the script (n/a: no price set).

## Repetition (scripts/line_variety.py)

| model | lang | n | near-dup | first6 | skel-dup | mean-sim | top opener |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | EN | 17 | 71% | 76% | 0% | 0.45 | 4/17 '<T> because it' |
| gpt-4.1-mini | ES | 17 | 71% | 71% | 0% | 0.50 | 4/17 '<T> porque iba' |
| gpt-4.1-mini | EL | 17 | 71% | 65% | 0% | 0.46 | 3/17 'η μπαλιά του' |
| gpt-6.1-sol | EN | 17 | 71% | 71% | 0% | 0.45 | 4/17 'it was <T>' |
| gpt-6.1-sol | ES | 17 | 76% | 65% | 0% | 0.50 | 4/17 'era <T> porque' |
| gpt-6.1-sol | EL | 17 | 76% | 76% | 0% | 0.48 | 4/17 'ήταν <T> επειδή' |

## Verifier reasons (retried and rejected moments)

### gpt-4.1-mini
- [41'] Weso Wendez (retried)
  - attempt 1: length: line_en has 25 words, max 24

### gpt-6.1-sol
- none

## English lines side by side

### [6'] Fenel Ravani | surprise_success | tricky | frame tier_first
- **gpt-4.1-mini**: Tricky because it was lofted high, yet Fenel Ravani's pass reaches a teammate.
- **gpt-6.1-sol**: It was tricky because it was lofted high, but Fenel Ravani's pass reaches a teammate.

### [6'] Quinus Morrley | surprise_success | difficult | frame reason
- **gpt-4.1-mini**: The pass from Quinus Morrley was difficult because it was lofted high, but it comes off.
- **gpt-6.1-sol**: The pass from Quinus Morrley was difficult because it was lofted high, but it comes off.

### [13'] Yanel Iverssen | surprise_success | tricky | frame concession
- **gpt-4.1-mini**: Yanel Iverssen's tricky pass finds a teammate, even though it was lofted high and under pressure.
- **gpt-6.1-sol**: Yanel Iverssen's tricky pass finds a teammate, even though it was lofted high and played under pressure.

### [14'] Marus Naskford | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini**: Even though it was played along the ground, Marus Naskford's easy pass does not reach a teammate.
- **gpt-6.1-sol**: Even though it was played along the ground and from deep, Marus Naskford's easy pass does not reach a teammate.

### [19'] Yanel Iverssen | surprise_success | very difficult | frame player_first
- **gpt-4.1-mini**: Yanel Iverssen plays a very difficult pass that reaches a teammate, even though it was lofted high and with a header.
- **gpt-6.1-sol**: Yanel Iverssen plays a very difficult pass that reaches a teammate, even though it was lofted high and played with a header.

### [19'] Alan Naskani | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini**: Easy because it was played along the ground, yet Alan Naskani's pass is lost.
- **gpt-6.1-sol**: It was easy because it was played along the ground, but Alan Naskani's pass is lost.

### [23'] Pelen Eldberg | surprise_success | difficult | frame reason
- **gpt-4.1-mini**: Pelen Eldberg's pass was difficult because it was lofted high, but it finds a teammate.
- **gpt-6.1-sol**: Pelen Eldberg's pass was difficult because it was lofted high, but it finds a teammate.

### [41'] Weso Wendez | surprise_failure | easy | frame concession
- **gpt-4.1-mini**: The pass from Weso Wendez is not completed, even though it came from deep and was played along the ground.
- **gpt-6.1-sol**: The easy pass from Weso Wendez does not reach a teammate, even though it came from deep along the ground.

### [44'] Jorar Wendard | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini**: Even though it was played along the ground, Jorar Wendard's easy pass does not find a teammate.
- **gpt-6.1-sol**: Even though it was played along the ground, Jorar Wendard's easy pass does not find a teammate.

### [46'] Alel Wendley | surprise_success | difficult | frame player_first
- **gpt-4.1-mini**: Alel Wendley plays a difficult pass that comes off, even though it was lofted high and under pressure.
- **gpt-6.1-sol**: Alel Wendley plays a difficult pass that comes off, even though it was lofted high and played under pressure.

### [52'] Jorar Wendard | surprise_success | difficult | frame tier_first
- **gpt-4.1-mini**: Difficult because it was lofted high, yet Jorar Wendard's pass finds a teammate.
- **gpt-6.1-sol**: It was difficult because it was lofted high, but Jorar Wendard's pass finds a teammate.

### [61'] Pelen Eldberg | surprise_failure | very easy | frame reason
- **gpt-4.1-mini**: The pass from Pelen Eldberg was very easy because it came from their own defensive third, but it does not reach a teammate.
- **gpt-6.1-sol**: The pass from Pelen Eldberg was very easy because it came from deep, but it does not reach a teammate.

### [63'] Coron Lundberg | surprise_success | tricky | frame concession
- **gpt-4.1-mini**: Coron Lundberg's tricky pass reaches a teammate, even though it was lofted high and under pressure.
- **gpt-6.1-sol**: Coron Lundberg's tricky pass reaches a teammate, even though it was lofted high and played under pressure.

### [78'] Pelen Eldberg | surprise_success | very difficult | frame concession_first
- **gpt-4.1-mini**: Even though it was lofted high and with a header, Pelen Eldberg's very difficult pass comes off.
- **gpt-6.1-sol**: Even though it was lofted high with a header, Pelen Eldberg's very difficult pass comes off.

### [80'] Weso Wendez | surprise_success | very difficult | frame player_first
- **gpt-4.1-mini**: Weso Wendez plays a very difficult pass that finds a teammate, even though it was lofted high and with a header.
- **gpt-6.1-sol**: Weso Wendez plays a very difficult pass that finds a teammate, even though it was lofted high and played with a header.

### [80'] Rosan Lundova | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini**: Easy because it was played along the ground, yet Rosan Lundova's pass does not reach a teammate.
- **gpt-6.1-sol**: It was easy because it went along the ground, but Rosan Lundova's pass does not reach a teammate.

### [81'] Fenel Ravani | surprise_failure | easy | frame reason
- **gpt-4.1-mini**: Fenel Ravani's pass was easy because it was played along the ground, but it does not find a teammate.
- **gpt-6.1-sol**: Fenel Ravani's pass was easy because it was played along the ground, but it does not find a teammate.

## Appendix: Spanish and Greek

### [6'] Fenel Ravani | surprise_success | tricky | frame tier_first
- **gpt-4.1-mini ES**: Complicado porque iba por alto, pero el pase de Fenel Ravani llega a un compañero.
- **gpt-6.1-sol ES**: Era complicado porque iba por alto, pero el pase de Fenel Ravani llega a un compañero.
- **gpt-4.1-mini EL**: Ήταν περίπλοκη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Fenel Ravani φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Ήταν περίπλοκη επειδή ήταν ψηλή, αλλά η πάσα του Fenel Ravani φτάνει σε συμπαίκτη.

### [6'] Quinus Morrley | surprise_success | difficult | frame reason
- **gpt-4.1-mini ES**: Un pase de Quinus Morrley era difícil porque iba por alto, pero sale bien.
- **gpt-6.1-sol ES**: Un pase de Quinus Morrley era difícil porque iba por alto, pero sale bien.
- **gpt-4.1-mini EL**: Η μπαλιά του Quinus Morrley ήταν δύσκολη λόγω του ότι ήταν ψηλή, αλλά πετυχαίνει.
- **gpt-6.1-sol EL**: Η μπαλιά του Quinus Morrley ήταν δύσκολη επειδή ήταν ψηλή, αλλά πετυχαίνει.

### [13'] Yanel Iverssen | surprise_success | tricky | frame concession
- **gpt-4.1-mini ES**: El pase complicado de Yanel Iverssen encuentra a un compañero, aunque iba por alto y bajo presión.
- **gpt-6.1-sol ES**: El pase complicado de Yanel Iverssen encuentra a un compañero, aunque iba por alto y se dio bajo presión.
- **gpt-4.1-mini EL**: Η περίπλοκη πάσα του Yanel Iverssen βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή και υπό πίεση.
- **gpt-6.1-sol EL**: Η περίπλοκη πάσα του Yanel Iverssen βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή και έγινε υπό πίεση.

### [14'] Marus Naskford | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini ES**: Aunque iba raso, el pase fácil de Marus Naskford no llega a un compañero.
- **gpt-6.1-sol ES**: Aunque iba raso y salía desde atrás, el pase fácil de Marus Naskford no llega a un compañero.
- **gpt-4.1-mini EL**: Παρόλο που ήταν συρτή, η εύκολη πάσα του Marus Naskford δεν φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Παρόλο που ήταν συρτή και έγινε από την άμυνα, η εύκολη πάσα του Marus Naskford δεν φτάνει σε συμπαίκτη.

### [19'] Yanel Iverssen | surprise_success | very difficult | frame player_first
- **gpt-4.1-mini ES**: Yanel Iverssen juega un pase muy difícil que llega a un compañero, aunque iba por alto y de cabeza.
- **gpt-6.1-sol ES**: Yanel Iverssen da un pase muy difícil que llega a un compañero, aunque iba por alto y era de cabeza.
- **gpt-4.1-mini EL**: Ο Yanel Iverssen παίζει μια πολύ δύσκολη πάσα που φτάνει σε συμπαίκτη, παρόλο που ήταν ψηλή και με κεφαλιά.
- **gpt-6.1-sol EL**: Ο Yanel Iverssen δίνει μια πολύ δύσκολη πάσα που φτάνει σε συμπαίκτη, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.

### [19'] Alan Naskani | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini ES**: Fácil porque iba raso, pero el pase de Alan Naskani se pierde.
- **gpt-6.1-sol ES**: Era fácil porque iba raso, pero el pase de Alan Naskani se pierde.
- **gpt-4.1-mini EL**: Εύκολη διότι ήταν συρτή, αλλά η πάσα του Alan Naskani χάνεται.
- **gpt-6.1-sol EL**: Ήταν εύκολη επειδή ήταν συρτή, αλλά η πάσα του Alan Naskani χάνεται.

### [23'] Pelen Eldberg | surprise_success | difficult | frame reason
- **gpt-4.1-mini ES**: El pase de Pelen Eldberg era difícil porque iba por alto, pero encuentra a un compañero.
- **gpt-6.1-sol ES**: El pase de Pelen Eldberg era difícil porque iba por alto, pero encuentra a un compañero.
- **gpt-4.1-mini EL**: Η πάσα του Pelen Eldberg ήταν δύσκολη λόγω του ότι ήταν ψηλή, αλλά βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Η πάσα του Pelen Eldberg ήταν δύσκολη επειδή ήταν ψηλή, αλλά βρίσκει συμπαίκτη.

### [41'] Weso Wendez | surprise_failure | easy | frame concession
- **gpt-4.1-mini ES**: Un pase de Weso Wendez no llega a un compañero, aunque salió desde atrás y fue raso.
- **gpt-6.1-sol ES**: Un pase fácil de Weso Wendez no llega a un compañero, aunque salió desde atrás y raso.
- **gpt-4.1-mini EL**: Η μπαλιά του Weso Wendez δεν φτάνει σε συμπαίκτη, παρόλο που έγινε από την άμυνα και ήταν συρτή.
- **gpt-6.1-sol EL**: Η εύκολη μπαλιά του Weso Wendez δεν φτάνει σε συμπαίκτη, παρόλο που ήρθε από την άμυνα και ήταν συρτή.

### [44'] Jorar Wendard | surprise_failure | easy | frame concession_first
- **gpt-4.1-mini ES**: Aunque iba raso, el pase fácil de Jorar Wendard no encuentra a un compañero.
- **gpt-6.1-sol ES**: Aunque iba raso, el pase fácil de Jorar Wendard no encuentra a un compañero.
- **gpt-4.1-mini EL**: Παρόλο που ήταν συρτή, η εύκολη πάσα του Jorar Wendard δεν βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Παρόλο που ήταν συρτή, η εύκολη πάσα του Jorar Wendard δεν βρίσκει συμπαίκτη.

### [46'] Alel Wendley | surprise_success | difficult | frame player_first
- **gpt-4.1-mini ES**: Alel Wendley realiza un pase difícil que sale bien, aunque iba por alto y bajo presión.
- **gpt-6.1-sol ES**: Alel Wendley da un pase difícil que sale bien, aunque iba por alto y se dio bajo presión.
- **gpt-4.1-mini EL**: Ο Alel Wendley παίζει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και υπό πίεση.
- **gpt-6.1-sol EL**: Ο Alel Wendley δίνει μια δύσκολη πάσα που πετυχαίνει, παρόλο που ήταν ψηλή και έγινε υπό πίεση.

### [52'] Jorar Wendard | surprise_success | difficult | frame tier_first
- **gpt-4.1-mini ES**: Difícil porque iba por alto, pero el pase de Jorar Wendard encuentra a un compañero.
- **gpt-6.1-sol ES**: Era difícil porque iba por alto, pero el pase de Jorar Wendard encuentra a un compañero.
- **gpt-4.1-mini EL**: Δύσκολη λόγω του ότι ήταν ψηλή, αλλά η πάσα του Jorar Wendard βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Ήταν δύσκολη επειδή ήταν ψηλή, αλλά η πάσα του Jorar Wendard βρίσκει συμπαίκτη.

### [61'] Pelen Eldberg | surprise_failure | very easy | frame reason
- **gpt-4.1-mini ES**: Un pase de Pelen Eldberg era muy fácil porque salía desde su propio tercio defensivo, pero no llega a un compañero.
- **gpt-6.1-sol ES**: Un pase de Pelen Eldberg era muy fácil porque venía desde atrás, pero no llega a un compañero.
- **gpt-4.1-mini EL**: Η μπαλιά του Pelen Eldberg ήταν πολύ εύκολη λόγω του ότι έγινε από την άμυνα, αλλά δεν φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Η μπαλιά του Pelen Eldberg ήταν πολύ εύκολη επειδή έγινε από την άμυνα, αλλά δεν φτάνει σε συμπαίκτη.

### [63'] Coron Lundberg | surprise_success | tricky | frame concession
- **gpt-4.1-mini ES**: El pase complicado de Coron Lundberg llega a un compañero, aunque iba por alto y bajo presión.
- **gpt-6.1-sol ES**: El pase complicado de Coron Lundberg llega a un compañero, aunque iba por alto y bajo presión.
- **gpt-4.1-mini EL**: Η περίπλοκη πάσα του Coron Lundberg φτάνει σε συμπαίκτη, παρόλο που ήταν ψηλή και υπό πίεση.
- **gpt-6.1-sol EL**: Η περίπλοκη πάσα του Coron Lundberg φτάνει σε συμπαίκτη, παρόλο που ήταν ψηλή και έγινε υπό πίεση.

### [78'] Pelen Eldberg | surprise_success | very difficult | frame concession_first
- **gpt-4.1-mini ES**: Aunque iba por alto y de cabeza, el pase muy difícil de Pelen Eldberg sale bien.
- **gpt-6.1-sol ES**: Aunque iba por alto y de cabeza, el pase muy difícil de Pelen Eldberg sale bien.
- **gpt-4.1-mini EL**: Παρόλο που ήταν ψηλή και με κεφαλιά, η πολύ δύσκολη πάσα του Pelen Eldberg πετυχαίνει.
- **gpt-6.1-sol EL**: Παρόλο που ήταν ψηλή και έγινε με κεφαλιά, η πολύ δύσκολη πάσα του Pelen Eldberg πετυχαίνει.

### [80'] Weso Wendez | surprise_success | very difficult | frame player_first
- **gpt-4.1-mini ES**: Weso Wendez juega un pase muy difícil que encuentra a un compañero, aunque iba por alto y de cabeza.
- **gpt-6.1-sol ES**: Weso Wendez da un pase muy difícil que encuentra a un compañero, aunque iba por alto y salió de cabeza.
- **gpt-4.1-mini EL**: Ο Weso Wendez παίζει μια πολύ δύσκολη πάσα που βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή και με κεφαλιά.
- **gpt-6.1-sol EL**: Ο Weso Wendez δίνει μια πολύ δύσκολη πάσα που βρίσκει συμπαίκτη, παρόλο που ήταν ψηλή και έγινε με κεφαλιά.

### [80'] Rosan Lundova | surprise_failure | easy | frame tier_first
- **gpt-4.1-mini ES**: Fácil porque iba raso, pero el pase de Rosan Lundova no llega a un compañero.
- **gpt-6.1-sol ES**: Era fácil porque iba raso, pero el pase de Rosan Lundova no llega a un compañero.
- **gpt-4.1-mini EL**: Εύκολη διότι ήταν συρτή, αλλά η πάσα του Rosan Lundova δεν φτάνει σε συμπαίκτη.
- **gpt-6.1-sol EL**: Ήταν εύκολη επειδή ήταν συρτή, αλλά η πάσα του Rosan Lundova δεν φτάνει σε συμπαίκτη.

### [81'] Fenel Ravani | surprise_failure | easy | frame reason
- **gpt-4.1-mini ES**: El pase de Fenel Ravani era fácil porque iba raso, pero no encuentra a un compañero.
- **gpt-6.1-sol ES**: El pase de Fenel Ravani era fácil porque iba raso, pero no encuentra a un compañero.
- **gpt-4.1-mini EL**: Η πάσα του Fenel Ravani ήταν εύκολη λόγω του ότι ήταν συρτή, αλλά δεν βρίσκει συμπαίκτη.
- **gpt-6.1-sol EL**: Η πάσα του Fenel Ravani ήταν εύκολη επειδή ήταν συρτή, αλλά δεν βρίσκει συμπαίκτη.
