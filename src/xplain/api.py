"""
Scoring API and the web page.

    uvicorn xplain.api:app --port 8000

POST /score takes one PassEvent and returns its xP (situational pass difficulty), difficulty,
band, delta and outlier flag. The scorer only ever receives ``event.inputs``.

GET / is the Studio page (src/xplain/web/index.html, its files under /static). It plays a match
bundle from GET /api/bundle/{name}: <bundle dir>/bundle_{name}.json, name matching ^[a-z0-9_]+$.
The bundle dir is create_app(bundle_dir=...), else XPLAIN_BUNDLE_DIR, else config.BUNDLE_DIR.

GET /api/live/health (live spike, live.py) is 404 unless XPLAIN_LIVE=1: one tiny model call with the app's managed
identity, at most once per 60 s server-wide (cached in between; Cache-Control: no-store).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analytics, live
from .analytics import Band, Outlier
from .config import BUNDLE_DIR
from .features import FEATURE_ORDER
from .schema import PassEvent
from .scoring import Scorer, XPassScorer, find_default_model

WEB_DIR = Path(__file__).resolve().parent / "web"
BUNDLE_NAME = r"^[a-z0-9_]+$"


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


def bundle_dir() -> Path:
    """Where match bundles are read from: XPLAIN_BUNDLE_DIR, else config.BUNDLE_DIR."""
    return Path(os.environ.get("XPLAIN_BUNDLE_DIR") or BUNDLE_DIR)


def create_app(scorer: Scorer | None = None, bundles: Path | None = None,
               live_check: live.LiveCheck | None = None) -> FastAPI:
    """Build the app. Without an injected scorer, the xPass model is loaded once at startup. live_check: the
    /api/live/health call (default live.LiveCheck(): no client is built until the first call)."""
    checker = live_check if live_check is not None else live.LiveCheck()

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

    @app.get("/api/live/health", include_in_schema=False)
    def live_health(response: Response) -> dict:
        if not live.enabled():
            raise HTTPException(status_code=404, detail="Not Found")
        response.headers["Cache-Control"] = "no-store"   # the server's cache is the only cache (age_s says how old)
        return checker.run()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    @app.get("/api/bundle/{name}")
    def match_bundle(name: Annotated[str, PathParam(pattern=BUNDLE_NAME, max_length=64)]) -> FileResponse:
        path = (bundles or bundle_dir()) / f"bundle_{name}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no bundle named {name!r}")
        return FileResponse(path, media_type="application/json")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()
