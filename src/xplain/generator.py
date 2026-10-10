"""
Seeded synthetic match generator (v0).

Produces a fictional but football-realistic match as a list of schema events, plus a separate
truth dict holding each player's hidden pass skill. Same seed and config -> identical output.
All randomness comes from one ``numpy.random.default_rng(seed)``, split with ``spawn`` into
independent streams for teams, play and player load.

Coordinates
    StatsBomb pitch, 120 x 80. Every event is in the frame of its own ``team_id``: that team
    attacks toward x=120 (goal centre (120, 40)). When the ball changes team it is mirrored,
    (x, y) -> (120 - x, 80 - y).

Teams
    Two clubs drawn from an invented list (config.CLUBS), 11 players each in a 4-4-2:
    1 GK, 4 DEF, 4 MID, 2 FWD. Names are built from seeded syllable lists and are unique
    within a match. The hidden pass skill ~ Normal(0, 1) goes to the truth file only.

Possession flow
    1. Each period starts with a Kick Off: home team in period 1, away team in period 2.
       Period 1 lasts 45 min + U(1, 4) min stoppage; period 2 starts right after the last
       period-1 event and lasts 45 min + U(2, 6) min. ``t`` runs on, halftime excluded.
    2. The team in possession acts with its ball holder. In open play the action may be a
       stoppage that restarts as a Free Kick (~1%; fouls are not modelled), a tackle by the
       defending team (~4%; won 60% -> turnover), a shot (beyond x=85, rising with x), and
       otherwise a pass.
    3. A pass starts where the ball is. Length ~ gamma (mean 18 m, long tail), direction ~
       von Mises around "forward", end clipped inside the pitch. Restarts (kick off, throw in,
       goal kick, corner, keeper) use their own shapes. Height depends on length; body part
       on height and passer. under_pressure ~ Bernoulli(0.12 + 0.20 * x / 120) in open play,
       with a pressure event 0.5-2.0 s before the pass by an opposing outfield player.
    4. Completion is the hidden logistic mechanism in config.py (skill, length, forward gain,
       pressure, height, restart bonus). It never uses the xPass model.
    5. A completed pass gives the ball to a teammate at the end point. A failed pass is a
       turnover at the mirrored end point, or (12%) goes out of play: its end is moved to the
       line it crossed and the other team restarts with a Throw In, Goal Kick or Corner.
    6. Shots: goal -> the conceding team kicks off; saved -> corner, or the keeper holds
       (Keeper pattern); off target -> corner (blocked), or goal kick.
    7. Next action time = event time + ball travel time + U(1, 4) s, plus a dead-ball delay
       before restarts and a loose-ball delay U(3, 8) s after an interception or won tackle.
    play_pattern is a possession attribute, as in StatsBomb: it is set when a possession starts
    (the restart type, or Regular Play after a turnover) and kept until the next restart or
    change of team. A possession_change event (team_id = gaining team, in its frame) marks
    every change of the team in possession.

Player load
    Every 300 s of match time, and at full time, one player_load event per player with
    cumulative distance, top speed so far and cumulative sprint count.

CLI
    python -m xplain.generator --seed 42 --out data/match_seed42.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import (
    CLUBS,
    FIRST_A,
    FIRST_B,
    FORMATION,
    GENERATOR_VERSION,
    LAST_C,
    LAST_D,
    POSITIONS,
    GenConfig,
)
from .schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    BodyPart,
    MatchMetaEvent,
    PassDetails,
    PassEvent,
    PassHeight,
    PassInputs,
    PlayerInfo,
    PlayerLoadEvent,
    PlayPattern,
    PossessionChangeEvent,
    PressureEvent,
    ShotEvent,
    TackleEvent,
)

GOAL_Y = PITCH_WIDTH / 2
POST_LOW_Y, POST_HIGH_Y = 36.0, 44.0
SET_PIECES = (PlayPattern.KICK_OFF, PlayPattern.GOAL_KICK, PlayPattern.CORNER, PlayPattern.FREE_KICK)


def _r2(v: float) -> float:
    return round(float(v), 2)


def _r3(v: float) -> float:
    return round(float(v), 3)


def _mirror(x: float, y: float) -> tuple[float, float]:
    return _r2(PITCH_LENGTH - x), _r2(PITCH_WIDTH - y)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _goal_angle(x: float, y: float) -> float:
    """Angle (radians) the goal mouth subtends from (x, y)."""
    dx = max(PITCH_LENGTH - x, 1e-6)
    return abs(math.atan2(POST_HIGH_Y - y, dx) - math.atan2(POST_LOW_Y - y, dx))


# ----------------------------------------------------------------------------- teams

def _make_teams(rng: np.random.Generator) -> list[dict]:
    picks = rng.choice(len(CLUBS), size=2, replace=False)
    used_names: set[str] = set()  # membership only, never iterated
    teams = []
    for k in picks:
        team_id, name = CLUBS[int(k)]
        numbers = sorted(int(n) for n in rng.choice(np.arange(2, 31), size=len(FORMATION) - 1, replace=False))
        players = []
        for i, pos in enumerate(FORMATION):
            while True:
                full = (
                    str(rng.choice(FIRST_A)) + str(rng.choice(FIRST_B)) + " "
                    + str(rng.choice(LAST_C)) + str(rng.choice(LAST_D))
                )
                if full not in used_names:
                    used_names.add(full)
                    break
            players.append({
                "player_id": f"{team_id}-p{i + 1}",
                "name": full,
                "position": pos,
                "shirt_number": 1 if pos == "GK" else numbers[i - 1],
                "team_id": team_id,
                "pass_skill": round(float(rng.normal(0.0, 1.0)), 4),
            })
        teams.append({"team_id": team_id, "name": name, "players": players})
    return teams


# ----------------------------------------------------------------------------- simulation

class _Match:
    def __init__(self, seed: int, cfg: GenConfig):
        self.seed = seed
        self.cfg = cfg
        self.match_id = f"m{seed}"
        team_rng, play_rng, load_rng = np.random.default_rng(seed).spawn(3)
        self.rng = play_rng
        self.load_rng = load_rng
        self.teams = _make_teams(team_rng)
        # records: (period, t, priority, index, model class, fields); seq is assigned after sorting
        self.records: list[tuple] = []
        self.last_t = 0.0
        self.period = 1
        # possession state, ball (x, y) in the frame of the team in possession
        self.poss = 0
        self.x, self.y = 60.0, 40.0
        self.holder: int | None = None
        self.pattern = PlayPattern.KICK_OFF
        self.restart = True
        self.t = 0.0

    # --- helpers ---------------------------------------------------------------------

    def _u(self, bounds: tuple[float, float]) -> float:
        return float(self.rng.uniform(bounds[0], bounds[1]))

    def _pid(self, team: int, idx: int) -> str:
        return self.teams[team]["players"][idx]["player_id"]

    def _tid(self, team: int) -> str:
        return self.teams[team]["team_id"]

    def _emit(self, cls, t: float, team: int, player: int | None, **fields) -> None:
        t = _r2(max(t, self.last_t))
        self.last_t = t
        self.records.append((
            self.period, t, 0, len(self.records), cls,
            dict(t=t, period=self.period, team_id=self._tid(team),
                 player_id=None if player is None else self._pid(team, player), **fields),
        ))

    def _pick(self, team: int, x: float, weights: tuple[float, ...] | None = None,
              exclude: int | None = None, outfield_only: bool = False) -> int:
        """Pick a player index, by position weights (zone weights from x if none given)."""
        if weights is None:
            weights = next(w for upper, w in self.cfg.zone_weights if x <= upper)
            if x > self.cfg.gk_max_x:
                weights = (0.0,) + tuple(weights[1:])
        counts = [FORMATION.count(p) for p in POSITIONS]
        w = np.array([
            0.0 if (i == exclude or (outfield_only and pos == "GK"))
            else weights[POSITIONS.index(pos)] / counts[POSITIONS.index(pos)]
            for i, pos in enumerate(FORMATION)
        ])
        if w.sum() <= 0:  # e.g. only the GK weighted and excluded
            w = np.array([0.0 if (i == exclude or pos == "GK") else 1.0 for i, pos in enumerate(FORMATION)])
        return int(self.rng.choice(len(FORMATION), p=w / w.sum()))

    def _set_possession(self, team: int, x: float, y: float, pattern: PlayPattern,
                        restart: bool, holder: int | None, t: float) -> None:
        self.poss, self.x, self.y = team, _r2(x), _r2(y)
        self.pattern, self.restart, self.holder, self.t = pattern, restart, holder, t

    def _change_possession(self, team: int, x: float, y: float, pattern: PlayPattern, restart: bool,
                           holder: int | None, event_t: float, next_t: float) -> None:
        self._emit(PossessionChangeEvent, event_t, team, holder, x=_r2(x), y=_r2(y),
                   gained_by_team_id=self._tid(team))
        self._set_possession(team, x, y, pattern, restart, holder, next_t)

    # --- period loop -----------------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        p1_end = cfg.half_length_s + self._u(cfg.stoppage_p1_s)
        self._play_period(1, 0.0, p1_end, kickoff_team=0)
        p2_start = _r2(max(p1_end, self.last_t))
        p2_end = p2_start + cfg.half_length_s + self._u(cfg.stoppage_p2_s)
        self._play_period(2, p2_start, p2_end, kickoff_team=1)
        self.p1_end = _r2(p1_end)
        self.p2_start = p2_start
        self.full_time = _r2(max(p2_end, self.last_t))

    def _play_period(self, period: int, start_t: float, end_t: float, kickoff_team: int) -> None:
        self.period = period
        if period > 1 and self.poss != kickoff_team:
            self._emit(PossessionChangeEvent, start_t, kickoff_team, None, x=60.0, y=40.0,
                       gained_by_team_id=self._tid(kickoff_team))
        self._set_possession(kickoff_team, 60.0, 40.0, PlayPattern.KICK_OFF, True, None, start_t)
        while self.t <= end_t:
            self._step()

    def _step(self) -> None:
        cfg, rng = self.cfg, self.rng
        if self.holder is None:
            self.holder = self._restart_taker()
        if not self.restart:
            if rng.random() < cfg.p_free_kick:
                self.pattern, self.restart = PlayPattern.FREE_KICK, True
                self.t += self._u(cfg.delay_free_kick_s)
                return
            if rng.random() < cfg.p_tackle:
                self._tackle()
                return
            if self.x > cfg.shot_min_x:
                p_shot = cfg.p_shot_base + cfg.p_shot_per_m * (self.x - cfg.shot_min_x)
                if rng.random() < p_shot:
                    self._shot()
                    return
        self._pass()

    def _restart_taker(self) -> int:
        p = self.pattern
        if p in (PlayPattern.GOAL_KICK, PlayPattern.KEEPER):
            return 0
        if p == PlayPattern.THROW_IN:
            return self._pick(self.poss, self.x, weights=(0.0, 0.6, 0.4, 0.0))
        if p == PlayPattern.CORNER:
            return self._pick(self.poss, self.x, weights=(0.0, 0.0, 0.7, 0.3))
        if p == PlayPattern.KICK_OFF:
            return self._pick(self.poss, self.x, weights=(0.0, 0.0, 0.5, 0.5))
        return self._pick(self.poss, self.x)

    # --- actions ---------------------------------------------------------------------

    def _tackle(self) -> None:
        cfg, rng = self.cfg, self.rng
        defending = 1 - self.poss
        tx, ty = _mirror(self.x, self.y)
        tackler = self._pick(defending, tx, outfield_only=True)
        won = bool(rng.random() < cfg.p_tackle_won)
        t = self.t
        self._emit(TackleEvent, t, defending, tackler, x=tx, y=ty, won=won)
        if won:
            self._change_possession(defending, tx, ty, PlayPattern.REGULAR, False, tackler,
                                    t, t + self._u(cfg.loose_ball_s) + self._u(cfg.decision_time_s))
        else:
            self.t = t + self._u(cfg.retain_after_tackle_s)

    def _shot(self) -> None:
        cfg, rng = self.cfg, self.rng
        team, x, y, t = self.poss, self.x, self.y, self.t
        dist = math.hypot(PITCH_LENGTH - x, GOAL_Y - y)
        angle = _goal_angle(x, y)
        a, b, c = cfg.on_target_coef
        on_target = bool(rng.random() < _sigmoid(a + b * dist + c * angle))
        a, b, c = cfg.goal_coef
        goal = on_target and bool(rng.random() < _sigmoid(a + b * dist + c * angle))
        speed = _r3(self._u(cfg.shot_speed_ms))
        self._emit(ShotEvent, t, team, self.holder, x=x, y=y, shot_speed_ms=speed,
                   on_target=on_target, goal=goal)
        arrive = t + dist / speed
        defending = 1 - team
        if goal:
            self._change_possession(defending, 60.0, 40.0, PlayPattern.KICK_OFF, True, None,
                                    arrive, arrive + self._u(cfg.delay_goal_s))
            return
        p_corner = cfg.p_corner_after_save if on_target else cfg.p_corner_after_miss
        if rng.random() < p_corner:
            corner_y = 0.0 if y < GOAL_Y else PITCH_WIDTH
            self._set_possession(team, PITCH_LENGTH, corner_y, PlayPattern.CORNER, True, None,
                                 arrive + self._u(cfg.delay_corner_s))
        elif on_target:  # keeper holds
            kx, ky = float(rng.uniform(4.0, 12.0)), float(rng.uniform(30.0, 50.0))
            self._change_possession(defending, kx, ky, PlayPattern.KEEPER, True, 0,
                                    arrive, arrive + self._u(cfg.delay_keeper_s))
        else:
            self._change_possession(defending, 6.0, float(rng.uniform(30.0, 50.0)), PlayPattern.GOAL_KICK,
                                    True, None, arrive, arrive + self._u(cfg.delay_goal_kick_s))

    def _design_pass(self, position: str, sx: float, sy: float) -> tuple[float, float, PassHeight, BodyPart]:
        """Return intended (length, direction, height, body part) from start-of-pass information."""
        cfg, rng = self.cfg, self.rng
        pattern = self.pattern if self.restart else PlayPattern.REGULAR
        lo, hi = cfg.pass_length_range

        def generic_length() -> float:
            return _clip(float(rng.gamma(cfg.pass_length_shape, cfg.pass_length_scale)), lo, hi)

        def generic_height(length: float) -> PassHeight:
            p_high = _clip(cfg.high_base + cfg.high_per_m * length, *cfg.high_range)
            u = rng.random()
            if u < p_high:
                return PassHeight.HIGH
            if u < p_high + cfg.p_low:
                return PassHeight.LOW
            return PassHeight.GROUND

        if pattern == PlayPattern.KICK_OFF:
            return float(rng.uniform(6.0, 18.0)), float(rng.vonmises(math.pi, 4.0)), PassHeight.GROUND, BodyPart.FOOT

        if pattern == PlayPattern.CORNER:
            if rng.random() < 0.75:  # delivered into the box
                tx, ty = float(rng.uniform(100.0, 114.0)), float(rng.uniform(30.0, 50.0))
                return math.hypot(tx - sx, ty - sy), math.atan2(ty - sy, tx - sx), PassHeight.HIGH, BodyPart.FOOT
            inward = math.pi * 0.75 if sy < GOAL_Y else -math.pi * 0.75
            return float(rng.uniform(5.0, 15.0)), float(rng.vonmises(inward, 4.0)), PassHeight.GROUND, BodyPart.FOOT

        if pattern == PlayPattern.GOAL_KICK:
            if rng.random() < 0.5:
                return float(rng.uniform(8.0, 20.0)), float(rng.vonmises(0.0, 1.0)), PassHeight.GROUND, BodyPart.FOOT
            length = _clip(float(rng.normal(55.0, 8.0)), 35.0, hi)
            return length, float(rng.vonmises(0.0, 3.0)), PassHeight.HIGH, BodyPart.FOOT

        if pattern == PlayPattern.THROW_IN:
            length = _clip(float(rng.gamma(2.0, 6.0)), 3.0, 35.0)
            inward = 0.6 if sy < GOAL_Y else -0.6
            return length, float(rng.vonmises(inward, 2.0)), generic_height(length), BodyPart.OTHER

        length = generic_length()
        direction = float(rng.vonmises(0.0, cfg.direction_kappa))
        if position == "GK" and rng.random() < cfg.p_keeper_arm:
            length = min(length, 40.0)
            height = PassHeight.GROUND if rng.random() < 0.7 else PassHeight.LOW
            return length, direction, height, BodyPart.KEEPER_ARM
        height = generic_height(length)
        body = BodyPart.FOOT
        if height == PassHeight.HIGH and position != "GK" and pattern == PlayPattern.REGULAR \
                and rng.random() < cfg.p_head_given_high:
            body = BodyPart.HEAD
        elif rng.random() < cfg.p_body_other:
            body = BodyPart.OTHER
        return length, direction, height, body

    def _ball_speed(self, length: float, height: PassHeight, body: BodyPart) -> float:
        cfg = self.cfg
        if body == BodyPart.KEEPER_ARM:
            base, per_m, sd, lo, hi = cfg.speed_keeper_arm
        elif height == PassHeight.HIGH:
            base, per_m, sd, lo, hi = cfg.speed_high
        elif height == PassHeight.LOW:
            base, per_m, sd, lo, hi = cfg.speed_low
        else:
            base, per_m, sd, lo, hi = cfg.speed_ground
        return _r3(_clip(base + per_m * length + float(self.rng.normal(0.0, sd)), lo, hi))

    def _restart_bonus(self) -> float:
        if not self.restart:
            return 0.0
        cfg = self.cfg
        return {
            PlayPattern.KICK_OFF: cfg.bonus_kick_off,
            PlayPattern.THROW_IN: cfg.bonus_throw_in,
            PlayPattern.GOAL_KICK: cfg.bonus_goal_kick,
            PlayPattern.FREE_KICK: cfg.bonus_free_kick,
            PlayPattern.CORNER: cfg.bonus_corner,
            PlayPattern.KEEPER: cfg.bonus_keeper,
        }.get(self.pattern, 0.0)

    def _pass(self) -> None:
        cfg, rng = self.cfg, self.rng
        team, passer, t = self.poss, self.holder, self.t
        sx, sy = self.x, self.y
        position = FORMATION[passer]
        player = self.teams[team]["players"][passer]

        length, direction, height, body = self._design_pass(position, sx, sy)
        m = cfg.end_margin
        ex = _r2(_clip(sx + length * math.cos(direction), m, PITCH_LENGTH - m))
        ey = _r2(_clip(sy + length * math.sin(direction), m, PITCH_WIDTH - m))
        length = _r2(math.hypot(ex - sx, ey - sy))

        set_piece = self.restart and self.pattern in SET_PIECES
        p_press = cfg.pressure_base + cfg.pressure_per_x * sx / PITCH_LENGTH
        under_pressure = (not set_piece) and bool(rng.random() < p_press)
        speed = self._ball_speed(length, height, body)

        # hidden completion mechanism: independent of any trained model
        logit = (
            cfg.intercept
            + cfg.w_skill * player["pass_skill"]
            + cfg.w_length * length
            + cfg.w_forward * max(ex - sx, 0.0)
            + cfg.w_pressure * under_pressure
            + cfg.w_high * (height == PassHeight.HIGH)
            + cfg.w_low * (height == PassHeight.LOW)
            + self._restart_bonus()
            # optional start-location terms, appended last so 0.0 weights leave the logit bit-identical
            + cfg.w_start_x * sx
            + cfg.w_dist_goal * math.hypot(PITCH_LENGTH - sx, GOAL_Y - sy)
            + cfg.w_angle_goal * math.atan2(GOAL_Y - sy, PITCH_LENGTH - sx)  # = features.angle_to_goal
        )
        completed = bool(rng.random() < _sigmoid(logit))

        recipient = None
        out_restart = None
        if completed:
            recipient = self._pick(team, ex, exclude=passer)
        elif rng.random() < cfg.p_out_of_play:
            # the ball leaves the pitch: move the end to the line it crossed
            if ex >= 112.0:
                ex, out_restart = PITCH_LENGTH, PlayPattern.GOAL_KICK
            elif ex <= 8.0:
                ex, out_restart = 0.0, PlayPattern.CORNER
            else:
                ey, out_restart = (0.0 if ey < GOAL_Y else PITCH_WIDTH), PlayPattern.THROW_IN
            length = _r2(math.hypot(ex - sx, ey - sy))

        if under_pressure:
            presser = self._pick(1 - team, PITCH_LENGTH - sx, outfield_only=True)
            px, py = _mirror(sx, sy)
            self._emit(PressureEvent, t - self._u(cfg.pressure_lead_s), 1 - team, presser,
                       x=px, y=py, pressed_player_id=self._pid(team, passer))
        self._emit(
            PassEvent, t, team, passer,
            inputs=PassInputs(start_x=sx, start_y=sy, height=height, body_part=body,
                              play_pattern=self.pattern, under_pressure=under_pressure),
            details=PassDetails(end_x=ex, end_y=ey, length=length, ball_speed_ms=speed, completed=completed,
                                recipient_id=None if recipient is None else self._pid(team, recipient)),
        )
        pass_t = self.last_t
        arrive = pass_t + (length / speed if speed > 0 else 0.0)
        decision = self._u(cfg.decision_time_s)
        other = 1 - team

        if completed:
            self._set_possession(team, ex, ey, self.pattern, False, recipient, arrive + decision)
        elif out_restart == PlayPattern.GOAL_KICK:
            self._change_possession(other, 6.0, float(rng.uniform(30.0, 50.0)), out_restart, True, None,
                                    arrive, arrive + self._u(cfg.delay_goal_kick_s))
        elif out_restart == PlayPattern.CORNER:
            corner_y = 0.0 if PITCH_WIDTH - ey < GOAL_Y else PITCH_WIDTH
            self._change_possession(other, PITCH_LENGTH, corner_y, out_restart, True, None,
                                    arrive, arrive + self._u(cfg.delay_corner_s))
        elif out_restart == PlayPattern.THROW_IN:
            nx, ny = _mirror(ex, ey)
            self._change_possession(other, nx, ny, out_restart, True, None,
                                    arrive, arrive + self._u(cfg.delay_throw_in_s))
        else:  # intercepted: turnover at the mirrored end point
            nx, ny = _mirror(ex, ey)
            gainer = self._pick(other, nx)
            self._change_possession(other, nx, ny, PlayPattern.REGULAR, False, gainer,
                                    arrive, arrive + self._u(cfg.loose_ball_s) + decision)

    # --- player load -----------------------------------------------------------------

    def add_player_load(self) -> None:
        cfg, rng = self.cfg, self.load_rng
        profiles = []
        for team in range(2):
            for idx, pos in enumerate(FORMATION):
                k = POSITIONS.index(pos)
                lo, hi = cfg.distance_per90_m[k]
                vmax_range = cfg.vmax_gk_ms if pos == "GK" else cfg.vmax_outfield_ms
                profiles.append({
                    "team": team, "idx": idx,
                    "rate": float(rng.uniform(lo, hi)) / 5400.0,
                    "vmax": min(float(rng.uniform(*vmax_range)), cfg.top_speed_cap_ms),
                    "sprint_rate": cfg.sprints_per90[k] / 5400.0,
                    "distance": 0.0, "top": 0.0, "sprints": 0,
                })

        checkpoints = []
        k = 1
        while k * cfg.load_interval_s < self.full_time:
            checkpoints.append(k * cfg.load_interval_s)
            k += 1
        checkpoints.append(self.full_time)

        prev = 0.0
        for cp in checkpoints:
            dt = cp - prev
            period = 1 if cp <= self.p2_start else 2
            for prof in profiles:
                prof["distance"] += prof["rate"] * dt * float(rng.uniform(*cfg.interval_noise))
                prof["top"] = max(prof["top"], prof["vmax"] * float(rng.uniform(0.75, 1.0)))
                prof["sprints"] += int(rng.poisson(prof["sprint_rate"] * dt))
                t = _r2(cp)
                self.records.append((
                    period, t, 1, len(self.records), PlayerLoadEvent,
                    dict(t=t, period=period, team_id=self._tid(prof["team"]),
                         player_id=self._pid(prof["team"], prof["idx"]),
                         distance_m=_r2(prof["distance"]), top_speed_ms=_r3(prof["top"]),
                         sprint_count=prof["sprints"]),
                ))
            prev = cp

    # --- output ------------------------------------------------------------------------

    def build_events(self) -> list:
        home, away = self.teams
        players = [
            PlayerInfo(**{k: p[k] for k in ("player_id", "name", "position", "shirt_number", "team_id")})
            for team in self.teams for p in team["players"]
        ]
        meta = dict(t=0.0, period=1, team_id=home["team_id"], player_id=None,
                    home_team_id=home["team_id"], away_team_id=away["team_id"],
                    home_name=home["name"], away_name=away["name"], seed=self.seed,
                    generator_version=GENERATOR_VERSION, players=players)
        records = [(1, 0.0, -1, -1, MatchMetaEvent, meta)] + sorted(self.records, key=lambda r: r[:4])
        return [
            cls(event_id=f"{self.match_id}-{seq:05d}", match_id=self.match_id, seq=seq, **fields)
            for seq, (_, _, _, _, cls, fields) in enumerate(records)
        ]

    def truth(self) -> dict:
        return {
            "seed": self.seed,
            "generator_version": GENERATOR_VERSION,
            "config": asdict(self.cfg),
            "periods": {"p1_end_t": self.p1_end, "p2_start_t": self.p2_start, "full_time_t": self.full_time},
            "players": [
                {"player_id": p["player_id"], "name": p["name"], "pass_skill": p["pass_skill"]}
                for team in self.teams for p in team["players"]
            ],
        }


def generate_match(seed: int, config: GenConfig = GenConfig()) -> tuple[list, dict]:
    """Generate one match. Returns (events ordered by seq, truth dict with the hidden skills)."""
    match = _Match(seed, config)
    match.run()
    match.add_player_load()
    return match.build_events(), match.truth()


def write_match(seed: int, out_path: str | Path, config: GenConfig = GenConfig()) -> tuple[Path, Path]:
    """Write events as JSONL (one event per line) and the truth file next to it."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    events, truth = generate_match(seed, config)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for ev in events:
            f.write(ev.model_dump_json() + "\n")
    truth_path = out.with_name(out.stem + ".truth.json")
    with truth_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(truth, indent=2) + "\n")
    return out, truth_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a seeded synthetic match (JSONL + truth file).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    events_path, truth_path = write_match(args.seed, args.out)
    print(f"wrote {events_path} and {truth_path}")


if __name__ == "__main__":
    main()
