"""Evaluation for the lending agent.

Six cases, each with an expected outcome. Prints per-case PASS/FAIL, then
task success rate and human-override rate.

Task success rate = cases where the actual outcome matched the expected one.
Human-override rate = cases that went through human approval, out of all
cases. A high override rate is not a failure on its own, it means the agent
is being conservative, but it is a number worth watching in production.

Run:  python eval.py
"""

from __future__ import annotations

from lending_agent import run_application

BASE = {
    "applicant_name": "Ayu Lestari",
    "applicant_id": "ID-1001",
    "monthly_income": 20_000,
    "loan_amount": 50_000,
    "loan_purpose": "home renovation",
    "employment_status": "employed",
    "existing_debt": 5_000,
}

CASES = [
    {
        "name": "normal approve",
        "raw": dict(BASE),
        "flags": {},
        "human": None,
        "expected_outcome": "disbursed",
        "expected_human": False,
        "why": "clean data, strong credit, small amount: auto-approve, no human",
    },
    {
        "name": "incomplete data",
        "raw": {k: v for k, v in BASE.items() if k != "monthly_income"},
        "flags": {},
        "human": None,
        "expected_outcome": "rejected_validation",
        "expected_human": False,
        "why": "missing monthly_income: intake validation fails loudly, nothing runs",
    },
    {
        "name": "kyc mismatch goes to human, human rejects",
        "raw": dict(BASE, applicant_id="ID-9999"),
        "flags": {},
        "human": {"approved": False, "reviewer": "arya", "note": "id not in registry"},
        "expected_outcome": "rejected_by_human",
        "expected_human": True,
        "why": "unknown id fails KYC, confidence drops, human rejects, no money moves",
    },
    {
        "name": "low credit score goes to human, human approves",
        "raw": dict(BASE, existing_debt=200_000),
        "flags": {},
        "human": {"approved": True, "reviewer": "arya", "note": "debt explained, docs verified"},
        "expected_outcome": "disbursed",
        "expected_human": True,
        "why": "heavy debt drops the score, agent routes to human instead of guessing",
    },
    {
        "name": "tool failure, retry then human",
        "raw": dict(BASE),
        "flags": {"kyc_fail_mode": "always"},
        "human": {"approved": False, "reviewer": "arya", "note": "KYC down, cannot verify"},
        "expected_outcome": "rejected_by_human",
        "expected_human": True,
        "why": "KYC tool fails twice, agent fails loudly to a human, never guesses",
    },
    {
        "name": "high amount needs human approval",
        "raw": dict(BASE, loan_amount=250_000),
        "flags": {},
        "human": {"approved": True, "reviewer": "arya", "note": "income docs verified"},
        "expected_outcome": "disbursed",
        "expected_human": True,
        "why": "amount above the auto-approve limit, human approves, then disburse",
    },
]


def main() -> int:
    passed = 0
    human_cases = 0
    print("lending agent eval, 6 cases")
    print("=" * 60)
    for i, case in enumerate(CASES, start=1):
        result = run_application(
            case["raw"],
            flags=case["flags"],
            human_decision=case["human"],
            thread_id=f"eval-{i}",
        )
        outcome_ok = result.get("outcome") == case["expected_outcome"]
        went_to_human = bool(result.get("human_decision"))
        human_ok = went_to_human == case["expected_human"]
        ok = outcome_ok and human_ok
        passed += 1 if ok else 0
        human_cases += 1 if went_to_human else 0
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {i}. {case['name']}")
        print(f"       expected: {case['expected_outcome']} (human={case['expected_human']})")
        print(f"       actual:   {result.get('outcome')} (human={went_to_human})")
        if result.get("error"):
            print(f"       error: {result['error']}")
        print(f"       why: {case['why']}")
    print("=" * 60)
    print(f"task success rate: {passed}/{len(CASES)} = {passed / len(CASES):.0%}")
    print(f"human-override rate: {human_cases}/{len(CASES)} = {human_cases / len(CASES):.0%}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
