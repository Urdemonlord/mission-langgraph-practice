# Lending agent practice project

A small, runnable LangGraph project for interview prep. It models a loan
application flowing through an agent: intake, KYC check, credit check,
decision, then either auto-approval or a human approval, then disbursement.

Everything external is a clearly labeled mock. There is no real LLM call, no
network, no API key, no real KYC data, and no real credit data. The "model"
is a small deterministic function that decides from the application data, so
every run is repeatable.

## How to run it

```bash
cd mission-langgraph-practice
python3 -m venv .venv
.venv/bin/pip install langgraph pydantic
.venv/bin/python lending_agent.py   # demo: one auto-approve, one human approval
.venv/bin/python eval.py            # eval: 6 cases, all should pass
```

## The graph

```
intake -> kyc_check -> credit_check -> assess -> decision
decision -> disburse (auto-approve)
decision -> human_approval -> disburse (human approved)
decision -> human_approval -> reject (human rejected)
```

State is a TypedDict (`LoanState`) carried through every node. Each node
appends to `state["trace"]`, a log of what it saw and what it returned, so
any run can be replayed and debugged step by step.

## Where the human approves

The `human_approval` node calls LangGraph `interrupt()`. The graph pauses
there and nothing proceeds until a human answers with approve or reject.
Three things route to a human:

1. Confidence below the threshold (0.75). The assess step was not sure.
2. Loan amount above the auto-approve limit (100,000). Big money always
   needs a person.
3. A tool failed after its retry. The agent never guesses when a check is
   unavailable, it fails loudly to a human instead.

The agent never rejects on its own. Rejection only comes from a human.
Disbursement only runs after an approval, auto or human, never before.

## How the confidence threshold works

The stub assess step starts at 0.9 and lowers confidence for each doubt:
KYC mismatch drops it to 0.30, a weak credit score to 0.45 or 0.65, a loan
above 6x monthly income to 0.60. Only a confidence of 0.75 or higher, with
a credit score of 650 or higher and an amount within the limit, gets an
auto-approve. Everything else is review, which means a human decides.

Start conservative, tune the threshold from reviewer feedback. A lower
threshold means more auto-approvals and more risk. A higher one means more
human review and slower throughput. That trade-off is the whole point.

## Files

- `lending_agent.py`: the graph, the nodes, the mock tools, the stub model,
  the retry logic, and a two-part demo.
- `eval.py`: 6 cases (normal approve, incomplete data, KYC mismatch, low
  credit score, tool failure, high amount). Prints per-case PASS/FAIL plus
  task success rate and human-override rate.

## What to be able to explain in the interview

- State: one TypedDict flows through the graph, every node reads and
  returns a slice of it, and the trace inside it makes runs replayable.
- Nodes and edges: each node does one job, and the routing logic lives in
  small functions (`route_after_assess`, `route_after_human`), not hidden
  inside the nodes.
- Human-in-the-loop: `interrupt()` pauses the graph, the checkpointer
  (MemorySaver) holds the state, and `Command(resume=...)` continues it
  with the human decision. In production this would be a durable
  checkpointer, not memory.
- Structured outputs: the assess step returns a Pydantic model, validated
  before anything acts on it. A bad parse fails loudly instead of silently
  passing wrong data.
- Tool failures: retry once with backoff, then fail loudly to a human.
  Never guess, never silently skip a check.
- Evaluation: an eval set of real cases, including edge cases, with task
  success rate, tool-call accuracy, and human-override rate. If the
  override rate climbs, something in the agent changed and you investigate.

## Honest limits

This is a learning project. It is not production ready. It does not deploy
to GCP or Vertex AI, it does not talk to real KYC or credit systems, and
the "model" is a rules function, not an LLM. Those are the next steps to
learn, not claims this project makes.
