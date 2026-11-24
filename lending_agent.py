"""Lending agent practice project.

A small LangGraph StateGraph for a loan application. The point is to learn
LangGraph patterns, not to build a real lending system. Every external check
here is a clearly labeled MOCK. No LLM call, no network, no API key: the
"model" is a deterministic function that decides from the application data.

Flow:
    intake -> kyc_check -> credit_check -> assess -> decision
    decision -> disburse (only after approval) or human_approval -> disburse
    human_approval uses LangGraph interrupt(), so a human decides.

Rules the graph enforces:
  - Invalid application data fails loudly at intake (Pydantic validation).
  - Structured output from the assess step is validated before any side effect.
  - A tool failure is retried once, then fails loudly to a human.
    The agent never guesses when a check is unavailable.
  - The agent never auto-approves when confidence is below the threshold
    or the amount is above the auto-approve limit. Those go to a human.
  - The agent never rejects on its own. Rejection only comes from a human.
  - Every node logs its inputs and outputs into state["trace"], so a run
    can be replayed and debugged step by step.

Run the demo:  python lending_agent.py
Run the eval:   python eval.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Config: thresholds. Start conservative, tune from reviewer feedback.
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75
AUTO_APPROVE_LIMIT = 100_000  # currency units; above this, a human must approve
MIN_CREDIT_SCORE_FOR_AUTO_APPROVE = 650

# ---------------------------------------------------------------------------
# Schemas. Structured outputs are validated here, before any side effect.
# ---------------------------------------------------------------------------


class LoanApplication(BaseModel):
    """The intake schema. Anything that does not fit fails loudly."""

    applicant_name: str = Field(min_length=1)
    applicant_id: str = Field(min_length=1)
    monthly_income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    loan_purpose: str = Field(min_length=1)
    employment_status: Literal["employed", "self-employed", "unemployed"]
    existing_debt: float = Field(ge=0)


class Assessment(BaseModel):
    """Structured output of the assess step. Validated before use."""

    decision: Literal["approve", "review"]
    confidence: float = Field(ge=0.0, le=1.0)
    credit_score: int = Field(ge=300, le=850)
    reasons: list[str]


class HumanDecision(BaseModel):
    """What a human reviewer returns when the graph is interrupted."""

    approved: bool
    reviewer: str = "human-reviewer"
    note: str = ""


# ---------------------------------------------------------------------------
# Mock tools. Clearly labeled mocks. No real KYC, no real credit data.
# ---------------------------------------------------------------------------

# Tiny in-memory registry the mock KYC check compares against.
KYC_REGISTRY = {
    "ID-1001": "Ayu Lestari",
    "ID-1002": "Budi Santoso",
    "ID-1003": "Citra Dewi",
}


class ToolFailure(Exception):
    """Raised when a mock tool fails after its retry."""


def _retry_once(fn, *args, **kwargs):
    """Call fn once. On failure, wait briefly and try exactly once more.
    If it still fails, raise ToolFailure instead of guessing."""
    try:
        return fn(*args, **kwargs)
    except Exception as first_error:
        time.sleep(0.2)
        try:
            return fn(*args, **kwargs)
        except Exception as second_error:
            raise ToolFailure(
                f"{fn.__name__} failed twice: {first_error}; then {second_error}"
            ) from second_error


def mock_kyc_check(application: LoanApplication, flags: dict) -> dict:
    """MOCK KYC check. Fails on demand when the test flags ask for it."""
    mode = flags.get("kyc_fail_mode")
    if mode == "always":
        raise RuntimeError("KYC provider unavailable (mock failure)")
    if mode == "once" and not flags.get("_kyc_failed_once"):
        flags["_kyc_failed_once"] = True
        raise RuntimeError("KYC provider timeout (mock failure)")

    expected_name = KYC_REGISTRY.get(application.applicant_id)
    if expected_name is None:
        return {"match": False, "reason": "applicant_id not found in registry"}
    if expected_name != application.applicant_name:
        return {"match": False, "reason": "name does not match registry"}
    return {"match": True, "reason": "name and id match registry"}


def mock_credit_check(application: LoanApplication, flags: dict) -> dict:
    """MOCK credit check. Score is derived deterministically from the
    application itself, so tests are repeatable with no external data."""
    mode = flags.get("credit_fail_mode")
    if mode == "always":
        raise RuntimeError("Credit bureau unavailable (mock failure)")
    if mode == "once" and not flags.get("_credit_failed_once"):
        flags["_credit_failed_once"] = True
        raise RuntimeError("Credit bureau timeout (mock failure)")

    yearly_income = application.monthly_income * 12
    dti = application.existing_debt / yearly_income if yearly_income else 1.0
    score = int(820 - 600 * dti)
    score = max(300, min(850, score))
    return {"credit_score": score, "debt_to_income": round(dti, 3)}


def mock_disburse(application: LoanApplication) -> dict:
    """MOCK disbursement. Only ever called after an approval."""
    return {
        "disbursed": True,
        "amount": application.loan_amount,
        "reference": f"DISB-{application.applicant_id}-{int(application.loan_amount)}",
    }


# ---------------------------------------------------------------------------
# Deterministic stub "model". Decides from data, no LLM, no network.
# ---------------------------------------------------------------------------


def stub_assess(application: LoanApplication, kyc: dict, credit: dict) -> Assessment:
    """Decide approve or review, with a confidence score.

    The stub only recommends approve when everything is strong. Anything
    doubtful becomes review, which routes to a human.
    """
    score = credit["credit_score"]
    reasons: list[str] = []
    confidence = 0.9

    if not kyc["match"]:
        reasons.append(f"KYC mismatch: {kyc['reason']}")
        confidence = min(confidence, 0.30)
    if score < MIN_CREDIT_SCORE_FOR_AUTO_APPROVE:
        reasons.append(f"credit score {score} below {MIN_CREDIT_SCORE_FOR_AUTO_APPROVE}")
        confidence = min(confidence, 0.45)
    elif score < 700:
        reasons.append(f"credit score {score} is middling")
        confidence = min(confidence, 0.65)
    if application.loan_amount > 6 * application.monthly_income:
        reasons.append("loan amount is more than 6x monthly income")
        confidence = min(confidence, 0.60)
    if application.loan_amount > AUTO_APPROVE_LIMIT:
        reasons.append(f"amount above auto-approve limit {AUTO_APPROVE_LIMIT}")
    if not reasons:
        reasons.append("KYC match, strong credit, amount within limits")

    decision = "approve" if confidence >= CONFIDENCE_THRESHOLD else "review"
    return Assessment(
        decision=decision,
        confidence=round(confidence, 2),
        credit_score=score,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# State and nodes. Every node appends to trace for replayability.
# ---------------------------------------------------------------------------


class LoanState(TypedDict, total=False):
    raw_application: dict
    application: dict
    flags: dict
    kyc: dict
    credit: dict
    assessment: dict
    needs_human: bool
    human_reason: str
    human_decision: dict
    disbursement: dict
    outcome: str
    error: str
    trace: list


def _log(state: LoanState, node: str, summary: dict) -> list:
    trace = list(state.get("trace") or [])
    trace.append(
        {
            "node": node,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": summary,
        }
    )
    return trace


def intake_node(state: LoanState) -> dict:
    """Validate the raw application. Invalid data fails loudly, here."""
    try:
        app = LoanApplication(**state["raw_application"])
    except ValidationError as exc:
        return {
            "error": f"intake validation failed: {exc.errors(include_url=False)}",
            "outcome": "rejected_validation",
            "trace": _log(state, "intake", {"status": "validation_failed"}),
        }
    return {
        "application": app.model_dump(),
        "trace": _log(
            state,
            "intake",
            {"status": "valid", "applicant": app.applicant_name, "amount": app.loan_amount},
        ),
    }


def kyc_node(state: LoanState) -> dict:
    """Run the mock KYC check. On tool failure after retry, route to human."""
    app = LoanApplication(**state["application"])
    flags = dict(state.get("flags") or {})
    try:
        result = _retry_once(mock_kyc_check, app, flags)
    except ToolFailure as exc:
        return {
            "error": str(exc),
            "needs_human": True,
            "human_reason": f"KYC tool failed after retry: {exc}",
            "flags": flags,
            "trace": _log(state, "kyc_check", {"status": "tool_failed_after_retry"}),
        }
    return {
        "kyc": result,
        "flags": flags,
        "trace": _log(state, "kyc_check", {"status": "ok", "match": result["match"]}),
    }


def credit_node(state: LoanState) -> dict:
    """Run the mock credit check. Same retry, then fail loudly to a human."""
    app = LoanApplication(**state["application"])
    flags = dict(state.get("flags") or {})
    try:
        result = _retry_once(mock_credit_check, app, flags)
    except ToolFailure as exc:
        return {
            "error": str(exc),
            "needs_human": True,
            "human_reason": f"credit tool failed after retry: {exc}",
            "flags": flags,
            "trace": _log(state, "credit_check", {"status": "tool_failed_after_retry"}),
        }
    return {
        "credit": result,
        "flags": flags,
        "trace": _log(
            state,
            "credit_check",
            {"status": "ok", "credit_score": result["credit_score"]},
        ),
    }


def assess_node(state: LoanState) -> dict:
    """Run the stub model and validate its structured output."""
    if "kyc" not in state or "credit" not in state:
        return {
            "error": "assess reached without KYC/credit results, refusing to decide",
            "needs_human": True,
            "human_reason": "assess ran without completed checks",
            "trace": _log(state, "assess", {"status": "missing_check_results"}),
        }
    app = LoanApplication(**state["application"])
    assessment = stub_assess(app, state["kyc"], state["credit"])
    needs_human = (
        assessment.decision != "approve"
        or assessment.confidence < CONFIDENCE_THRESHOLD
        or app.loan_amount > AUTO_APPROVE_LIMIT
    )
    reason = "; ".join(assessment.reasons)
    return {
        "assessment": assessment.model_dump(),
        "needs_human": needs_human,
        "human_reason": reason if needs_human else "",
        "trace": _log(
            state,
            "assess",
            {
                "status": "ok",
                "decision": assessment.decision,
                "confidence": assessment.confidence,
                "credit_score": assessment.credit_score,
            },
        ),
    }


def human_approval_node(state: LoanState) -> dict:
    """Pause the graph and ask a human. Nothing proceeds until they answer."""
    decision = interrupt(
        {
            "reason": state.get("human_reason") or state.get("error") or "review needed",
            "application": state.get("application"),
            "assessment": state.get("assessment"),
        }
    )
    human = HumanDecision(**decision)
    return {
        "human_decision": human.model_dump(),
        "trace": _log(
            state,
            "human_approval",
            {"status": "decided", "approved": human.approved, "by": human.reviewer},
        ),
    }


def disburse_node(state: LoanState) -> dict:
    """Disburse. Only reachable after an approval, never on a guess."""
    app = LoanApplication(**state["application"])
    result = mock_disburse(app)
    return {
        "disbursement": result,
        "outcome": "disbursed",
        "trace": _log(
            state,
            "disburse",
            {"status": "ok", "reference": result["reference"]},
        ),
    }


def reject_node(state: LoanState) -> dict:
    """End the run without disbursing. Only a human decision reaches here."""
    return {
        "outcome": "rejected_by_human",
        "trace": _log(state, "reject", {"status": "ok"}),
    }


# ---------------------------------------------------------------------------
# Routing. The decision logic lives here, in one place.
# ---------------------------------------------------------------------------


def route_after_assess(state: LoanState) -> str:
    if state.get("error"):
        return "human_approval"
    if state.get("needs_human"):
        return "human_approval"
    return "disburse"


def route_after_human(state: LoanState) -> str:
    decision = state.get("human_decision") or {}
    if decision.get("approved"):
        return "disburse"
    return "reject"


def route_after_intake(state: LoanState) -> str:
    if state.get("error"):
        return END
    return "kyc_check"


def route_after_check(state: LoanState) -> str:
    """After a check node: on tool failure, skip straight to a human.
    Never let a later node run on a missing check result."""
    if state.get("error"):
        return "human_approval"
    return "next"


def build_graph() -> Any:
    graph = StateGraph(LoanState)
    graph.add_node("intake", intake_node)
    graph.add_node("kyc_check", kyc_node)
    graph.add_node("credit_check", credit_node)
    graph.add_node("assess", assess_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("disburse", disburse_node)
    graph.add_node("reject", reject_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", route_after_intake, {"kyc_check": "kyc_check", END: END})
    graph.add_conditional_edges(
        "kyc_check",
        route_after_check,
        {"human_approval": "human_approval", "next": "credit_check"},
    )
    graph.add_conditional_edges(
        "credit_check",
        route_after_check,
        {"human_approval": "human_approval", "next": "assess"},
    )
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {"human_approval": "human_approval", "disburse": "disburse"},
    )
    graph.add_conditional_edges(
        "human_approval",
        route_after_human,
        {"disburse": "disburse", "reject": "reject"},
    )
    graph.add_edge("disburse", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=MemorySaver())


def run_application(
    raw_application: dict,
    flags: Optional[dict] = None,
    human_decision: Optional[dict] = None,
    thread_id: str = "demo",
) -> dict:
    """Run one application through the graph.

    If the graph pauses for a human and human_decision is given, resume with
    it. If it pauses and no human_decision is given, return the paused state
    so the caller can see what the human was asked.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: LoanState = {
        "raw_application": raw_application,
        "flags": dict(flags or {}),
        "trace": [],
    }
    result = graph.invoke(initial, config)
    if "__interrupt__" in result and result["__interrupt__"]:
        if human_decision is None:
            return result
        # Resume on the SAME graph object, so the checkpointer still holds
        # the paused state. A fresh graph would have an empty checkpointer.
        result = graph.invoke(Command(resume=human_decision), config)
    return result


def _print_result(result: dict) -> None:
    print(f"outcome: {result.get('outcome')}")
    if result.get("assessment"):
        a = result["assessment"]
        print(f"assessment: {a['decision']} confidence={a['confidence']} score={a['credit_score']}")
        print(f"reasons: {'; '.join(a['reasons'])}")
    if result.get("human_decision"):
        print(f"human decision: {result['human_decision']}")
    if result.get("disbursement"):
        print(f"disbursement: {result['disbursement']}")
    if result.get("error"):
        print(f"error: {result['error']}")
    print("trace:")
    for entry in result.get("trace") or []:
        print(f"  - {entry['node']}: {json.dumps(entry['summary'])}")


if __name__ == "__main__":
    print("=== Demo 1: clean application, auto-approved ===")
    good = {
        "applicant_name": "Ayu Lestari",
        "applicant_id": "ID-1001",
        "monthly_income": 20_000,
        "loan_amount": 50_000,
        "loan_purpose": "home renovation",
        "employment_status": "employed",
        "existing_debt": 5_000,
    }
    _print_result(run_application(good, thread_id="demo-1"))

    print()
    print("=== Demo 2: high amount, human approval required ===")
    big = dict(good, loan_amount=250_000)
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo-2"}}
    paused = graph.invoke({"raw_application": big, "flags": {}, "trace": []}, config)
    print("paused for human approval, asked:")
    print(json.dumps(paused["__interrupt__"][0].value, indent=2))
    resumed = graph.invoke(
        Command(
            resume={"approved": True, "reviewer": "arya", "note": "verified income docs"}
        ),
        config,
    )
    _print_result(resumed)
