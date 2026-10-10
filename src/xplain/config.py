"""
Constants for the synthetic match generator.

Every tunable number lives in ``GenConfig`` so the truth file can record the exact settings a
match was generated with. The pass-completion mechanism is hand-set and independent of the xPass
model. Only its intercept was adjusted (2.6 -> 2.9), so the overall completion rate lands in a
realistic 78-86% band.

Final pass-completion mechanism (hidden, never written to the events file):

    logit = 2.9
            + 0.45  * pass_skill                (player latent, Normal(0, 1))
            - 0.045 * length_m
            - 0.03  * max(forward_gain_m, 0)
            - 0.55  * under_pressure
            - 0.35  * is_high
            - 0.25  * is_low
            + restart_bonus                     (first pass of a restart only)
    completed ~ Bernoulli(sigmoid(logit))
"""
from __future__ import annotations

from dataclasses import dataclass

GENERATOR_VERSION = "0.1.0"

# Invented clubs as (team_id, name). Checked against web search: none is a real club.
CLUBS: tuple[tuple[str, str], ...] = (
    ("brk", "Brackmoor Athletic"),
    ("orr", "Orrinford Albion"),
    ("dsk", "Duskmere United"),
    ("wrv", "Wrenvale Town"),
    ("kld", "Kelderwick City"),
    ("mrf", "Marrowfield Wanderers"),
)

# Syllables for fictional player names (first = A + B, last = C + D).
FIRST_A: tuple[str, ...] = (
    "Al", "Bren", "Cor", "Dav", "El", "Fen", "Gar", "Hal", "Ish", "Jor", "Kel", "Lo",
    "Mar", "Nev", "Or", "Pel", "Quin", "Ros", "Sav", "Tor", "Ul", "Vin", "Wes", "Yan", "Zel",
)
FIRST_B: tuple[str, ...] = ("an", "en", "io", "o", "as", "ek", "ir", "on", "us", "ar", "el", "ian")
LAST_C: tuple[str, ...] = (
    "Ash", "Brin", "Cal", "Dorn", "Eld", "Fross", "Garr", "Holm", "Ivers", "Kest", "Lund",
    "Morr", "Nask", "Ost", "Pell", "Rav", "Stav", "Thal", "Varn", "Wend",
)
LAST_D: tuple[str, ...] = ("ley", "ford", "sen", "ard", "wick", "mont", "berg", "ani", "ova", "ez", "ton", "ic")

# Team shape (4-4-2), in player order: p1 is the goalkeeper.
FORMATION: tuple[str, ...] = ("GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "FWD", "FWD")
POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class GenConfig:
    # --- clock (seconds) ---
    half_length_s: float = 2700.0
    stoppage_p1_s: tuple[float, float] = (60.0, 240.0)
    stoppage_p2_s: tuple[float, float] = (120.0, 360.0)
    decision_time_s: tuple[float, float] = (1.0, 4.0)
    retain_after_tackle_s: tuple[float, float] = (0.5, 2.0)
    loose_ball_s: tuple[float, float] = (3.0, 8.0)   # extra time after an interception or won tackle
    # dead-ball delays before the restart pass
    delay_throw_in_s: tuple[float, float] = (15.0, 30.0)
    delay_goal_kick_s: tuple[float, float] = (25.0, 45.0)
    delay_corner_s: tuple[float, float] = (25.0, 45.0)
    delay_free_kick_s: tuple[float, float] = (30.0, 60.0)
    delay_keeper_s: tuple[float, float] = (5.0, 12.0)
    delay_goal_s: tuple[float, float] = (60.0, 100.0)

    # --- pass design ---
    pass_length_shape: float = 2.0          # gamma: mean = shape * scale = 18 m
    pass_length_scale: float = 9.0
    pass_length_range: tuple[float, float] = (2.0, 75.0)
    direction_kappa: float = 0.9            # von Mises concentration around "forward"
    end_margin: float = 0.5                 # in-play pass ends are kept this far inside the lines
    p_low: float = 0.10
    high_base: float = -0.06                # p(high) = clip(high_base + high_per_m * length)
    high_per_m: float = 0.009
    high_range: tuple[float, float] = (0.02, 0.85)
    p_head_given_high: float = 0.2          # outfield open-play high passes
    p_keeper_arm: float = 0.35              # goalkeeper passes other than goal kicks
    p_body_other: float = 0.01
    # ball speed = clip(base + per_m * length + Normal(0, sd), lo, hi): (base, per_m, sd, lo, hi)
    speed_ground: tuple[float, float, float, float, float] = (9.0, 0.30, 1.5, 10.0, 22.0)
    speed_low: tuple[float, float, float, float, float] = (11.0, 0.30, 1.5, 12.0, 24.0)
    speed_high: tuple[float, float, float, float, float] = (12.0, 0.25, 2.0, 14.0, 26.0)
    speed_keeper_arm: tuple[float, float, float, float, float] = (7.0, 0.25, 1.0, 8.0, 16.0)

    # --- who passes / receives: weights for (GK, DEF, MID, FWD) by ball x ---
    zone_weights: tuple[tuple[float, tuple[float, float, float, float]], ...] = (
        (30.0, (0.15, 0.55, 0.25, 0.05)),
        (80.0, (0.0, 0.30, 0.50, 0.20)),
        (120.0, (0.0, 0.10, 0.45, 0.45)),
    )
    gk_max_x: float = 25.0                  # goalkeeper only involved below this x in open play

    # --- pressure ---
    pressure_base: float = 0.12             # p = base + per_x * x / 120
    pressure_per_x: float = 0.20
    pressure_lead_s: tuple[float, float] = (0.5, 2.0)

    # --- hidden pass-completion mechanism (see module docstring) ---
    intercept: float = 2.9
    w_skill: float = 0.45
    w_length: float = -0.045
    w_forward: float = -0.03
    w_pressure: float = -0.55
    w_high: float = -0.35
    w_low: float = -0.25
    bonus_kick_off: float = 1.5
    bonus_throw_in: float = 0.3
    bonus_goal_kick: float = 0.0
    bonus_free_kick: float = 0.5
    bonus_corner: float = -0.5
    bonus_keeper: float = 0.5

    # --- turnovers and restarts ---
    p_out_of_play: float = 0.12             # share of failed passes that leave the pitch
    p_free_kick: float = 0.01               # per open-play action: stoppage -> Free Kick
    p_tackle: float = 0.04                  # per open-play action
    p_tackle_won: float = 0.6

    # --- shots ---
    shot_min_x: float = 85.0
    p_shot_base: float = 0.035             # per action beyond shot_min_x: base + per_m * (x - min_x)
    p_shot_per_m: float = 0.002
    shot_speed_ms: tuple[float, float] = (18.0, 35.0)
    # on target: sigmoid(a + b * dist_m + c * goal_angle_rad)
    on_target_coef: tuple[float, float, float] = (0.3, -0.05, 0.8)
    # goal given on target: same form
    goal_coef: tuple[float, float, float] = (0.3, -0.09, 1.5)
    p_corner_after_save: float = 0.3
    p_corner_after_miss: float = 0.2

    # --- player load ---
    load_interval_s: float = 300.0
    # distance per 90 minutes in metres, uniform range by position (GK, DEF, MID, FWD)
    distance_per90_m: tuple[tuple[float, float], ...] = (
        (4000.0, 5000.0), (9500.0, 10800.0), (10500.0, 12000.0), (9000.0, 10500.0),
    )
    interval_noise: tuple[float, float] = (0.85, 1.15)
    vmax_gk_ms: tuple[float, float] = (6.5, 8.0)
    vmax_outfield_ms: tuple[float, float] = (8.6, 10.5)
    top_speed_cap_ms: float = 10.5
    sprints_per90: tuple[float, float, float, float] = (0.5, 25.0, 30.0, 38.0)


@dataclass(frozen=True)
class AnalyticsConfig:
    """Thresholds for turning xP into display labels.

    All values are provisional: tune them from the observed xP distribution.
    """

    # difficulty = round(100 * (1 - xP)); band lower bounds (provisional, tune from the xP distribution)
    band_moderate_from: int = 20
    band_hard_from: int = 40
    band_exceptional_from: int = 60
    # outlier flags (provisional, tune from the xP distribution)
    surprise_success_max_xp: float = 0.35   # completed although xP <= this
    surprise_failure_min_xp: float = 0.92   # failed although xP >= this
