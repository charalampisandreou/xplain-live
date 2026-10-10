"""
Scoring API.

    uvicorn xplain.api:app --port 8000

POST /score takes one PassEvent and returns its xP (situational pass difficulty), difficulty,
band, delta and outlier flag. The scorer only ever receives ``event.inputs``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from . import analytics
from .analytics import Band, Outlier
from .features import FEATURE_ORDER
from .schema import PassEvent
from .scoring import Scorer, XPassScorer, find_default_model


class ScoreResponse(BaseModel):
    event_id: str
    match_id: str
    seq: int
    t: float
    xp: float
    difficulty: int
    difficulty_band: Band
    completed: bool
    delta: float
    outlier: Outlier | None
    drivers: list
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int


def create_app(scorer: Scorer | None = None) -> FastAPI:
    """Build the app. Without an injected scorer, the xPass model is loaded once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = scorer if scorer is not None else XPassScorer(*find_default_model())
        yield

    app = FastAPI(title="xplain-live scoring", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model_version=request.app.state.scorer.model_version,
                              n_features=len(FEATURE_ORDER))

    @app.post("/score", response_model=ScoreResponse)
    def score(event: PassEvent, request: Request) -> ScoreResponse:
        result = request.app.state.scorer.score([event.inputs])[0]
        xp = result.expected_probability
        completed = event.details.completed  # outcome is used for delta and flags, never for xP
        d = analytics.difficulty(xp)
        return ScoreResponse(
            event_id=event.event_id,
            match_id=event.match_id,
            seq=event.seq,
            t=event.t,
            xp=xp,
            difficulty=d,
            difficulty_band=analytics.difficulty_band(d),
            completed=completed,
            delta=analytics.delta(completed, xp),
            outlier=analytics.outlier(completed, xp),
            drivers=list(result.drivers),
            model_version=request.app.state.scorer.model_version,
        )

    return app


app = create_app()
