"""
FastAPI server for the grocery agent web UI.

Wraps `agent.loop.chat()` behind a small JSON API and serves the
single-page frontend from `web/`. In-memory session storage keyed by a
UUID the client keeps in localStorage; good enough for local dev / demo,
swap for Redis if you ever need to run more than one worker.

Run:
    uv run uvicorn server:app --reload --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv

# Load .env BEFORE importing anything that reads config.settings, so the
# LLM client sees OPENROUTER_API_KEY / GOOGLE_API_KEY / USE_OPENROUTER etc.
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent.loop import chat  # noqa: E402
from agent.state import AgentState  # noqa: E402
from config.settings import HOME_ADDRESS, HOME_LAT, HOME_LNG  # noqa: E402

log = logging.getLogger(__name__)

WEB_DIR = ROOT / "web"


# ─────────────────────────── session store ───────────────────────────


class SessionStore:
    """Thread-safe in-memory map of session_id -> AgentState."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> AgentState:
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                st = AgentState()
                self._sessions[session_id] = st
            return st

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = AgentState()

    def peek(self, session_id: str) -> AgentState | None:
        with self._lock:
            return self._sessions.get(session_id)


SESSIONS = SessionStore()


# ─────────────────────────── state → web view ────────────────────────


def build_web_state(state: AgentState) -> dict[str, Any]:
    """Flatten AgentState into a JSON shape the frontend can render
    without any extra joins: each route stop carries its coordinates
    AND the items to buy there, so the map popups / list cards are
    one-to-one with what the user sees."""
    plan_wrap = state.shopping_plan or {}
    plan = plan_wrap.get("plan") or {}
    stores_meta = plan_wrap.get("stores_meta") or {}

    by_store = [
        {
            "store_id": sid,
            "store_name": (stores_meta.get(sid) or {}).get("name") or sid,
            "store_branch": (stores_meta.get(sid) or {}).get("branch"),
            "address": (stores_meta.get(sid) or {}).get("address"),
            "lat": (stores_meta.get(sid) or {}).get("lat"),
            "lng": (stores_meta.get(sid) or {}).get("lng"),
            "items": [
                {
                    "item": e.get("item"),
                    "price": e.get("price"),
                    "url": e.get("url"),
                    "source_item": e.get("source_item"),
                }
                for e in entries
            ],
            "subtotal": round(sum(float(e.get("price") or 0) for e in entries), 2),
        }
        for sid, entries in plan.items()
    ]

    shopping_plan_view: dict | None = None
    if plan:
        shopping_plan_view = {
            "total_cost": plan_wrap.get("total_cost"),
            "not_found": plan_wrap.get("not_found") or [],
            "unfulfilled_preferences": plan_wrap.get("unfulfilled_preferences") or [],
            "by_store": by_store,
        }

    route_view: dict | None = None
    route = state.route_plan or None
    if route:
        stops = []
        for s in route.get("ordered_stops") or []:
            sid = s.get("store_id") or ""
            kind = s.get("kind") or "store"
            lat = lng = None
            items_here: list[dict] = []
            if kind == "store":
                meta = stores_meta.get(sid) or {}
                lat = meta.get("lat")
                lng = meta.get("lng")
                items_here = [
                    {
                        "item": e.get("item"),
                        "price": e.get("price"),
                        "url": e.get("url"),
                    }
                    for e in (plan.get(sid) or [])
                ]
            else:
                # Destination — match by label (ordered_stops.name)
                for d in state.destinations:
                    if (d.get("label") or "") == (s.get("name") or ""):
                        lat = d.get("lat")
                        lng = d.get("lng")
                        break
            stops.append({
                "kind": kind,
                "store_id": sid,
                "name": s.get("name"),
                "address": s.get("address"),
                "lat": lat,
                "lng": lng,
                "leg_duration_min": s.get("leg_duration_min"),
                "leg_distance_km": s.get("leg_distance_km"),
                "items": items_here,
            })
        home_view = route.get("home") or (
            {"lat": state.home.get("lat"),
             "lng": state.home.get("lng"),
             "address": state.home.get("address")}
            if state.home
            else {"lat": HOME_LAT, "lng": HOME_LNG, "address": HOME_ADDRESS}
        )
        route_view = {
            "ordered_stops": stops,
            "home": home_view,
            "total_duration_min": route.get("total_duration_min"),
            "total_distance_km": route.get("total_distance_km"),
            "ors_directions_url": route.get("ors_directions_url"),
            "destinations_count": route.get("destinations_count", 0),
        }

    return {
        "raw_items": [
            {
                "name": it.get("name"),
                "quantity": it.get("quantity"),
                "unit": it.get("unit"),
                "ambiguous": bool(it.get("ambiguous", False)),
            }
            for it in state.raw_items
        ],
        "shopping_plan": shopping_plan_view,
        "route": route_view,
        "destinations": [
            {"label": d.get("label"), "address": d.get("address"),
             "lat": d.get("lat"), "lng": d.get("lng")}
            for d in state.destinations
        ],
        "preferences": state.preferences,
        "preferred_stores": state.preferred_stores,
        "want_errand": state.want_errand,
        "errand_quote": state.errand_quote,
        "pending_dish": (
            {
                "name": state.pending_dish.get("name"),
                "cuisine": state.pending_dish.get("cuisine"),
                "ingredients": [
                    {"name": i.get("name"), "quantity": i.get("quantity"),
                     "unit": i.get("unit"), "pantry": bool(i.get("pantry"))}
                    for i in (state.pending_dish.get("ingredients") or [])
                ],
            }
            if state.pending_dish else None
        ),
        "last_options": state.last_options,
    }


# ─────────────────────────── schemas ────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    state: dict


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


# ─────────────────────────── app ────────────────────────────────────


app = FastAPI(title="Grocery Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    state = SESSIONS.get_or_create(req.session_id)
    try:
        reply = chat(state, req.message)
    except Exception as e:
        log.exception("chat() failed")
        raise HTTPException(status_code=500, detail=f"agent error: {e}") from e
    return ChatResponse(reply=reply, state=build_web_state(state))


@app.get("/api/state")
def api_state(session_id: str) -> dict:
    state = SESSIONS.peek(session_id)
    if state is None:
        # A fresh session — return an empty view so the frontend can render.
        return {"state": build_web_state(AgentState())}
    return {"state": build_web_state(state)}


@app.post("/api/reset")
def api_reset(req: ResetRequest) -> dict:
    SESSIONS.reset(req.session_id)
    return {"ok": True, "session_id": req.session_id,
            "state": build_web_state(AgentState())}


@app.get("/api/new-session")
def api_new_session() -> dict:
    """Give the client a fresh session_id. Optional — clients can also
    generate their own UUID in JS; this just keeps server and client in
    sync if the server restarts."""
    return {"session_id": str(uuid.uuid4())}


# ─────────────────────────── static site ────────────────────────────


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(idx)
