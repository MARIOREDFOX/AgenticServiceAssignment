"""
run_agent.py
-------------
Runs the AI assignment agent against your REAL ServiceNow developer instance.
Shows every decision live with colour-coded output and a running summary.

Usage:
    python run_agent.py              # process all unassigned tickets once
    python run_agent.py --watch      # keep polling every 30s (live mode)
    python run_agent.py --reset      # re-open all assigned tickets (for re-testing)
    python run_agent.py --status     # show current ticket states only, don't process
"""

import argparse
import logging
import os
import sys
import time

import requests
import yaml

# ── Colour codes ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# Suppress low-level logs so output stays clean
logging.basicConfig(level=logging.WARNING)

# Make agents importable
sys.path.insert(0, os.path.dirname(__file__))

from agents.ingestion_agent       import TicketIngestionAgent
from agents.knowledge_agent       import KnowledgeAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.prediction_agent      import AssignmentPredictionAgent
from agents.confidence_engine     import ConfidenceScoringEngine
from agents.decision_agent        import DecisionAgent
from agents.servicenow_agent      import ServiceNowUpdateAgent
from agents.learning_agent        import LearningAgent

# ── Load config ───────────────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

INSTANCE = config["servicenow"]["instance_url"].rstrip("/")
AUTH     = (config["servicenow"]["username"], config["servicenow"]["password"])
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}


# ── Visual helpers ────────────────────────────────────────────────────────────
def confidence_bar(score: float) -> str:
    filled = int(score)
    empty  = 10 - filled
    color  = GREEN if score > 7 else YELLOW if score > 5 else RED
    return f"{color}{'█' * filled}{'░' * empty}{RESET} {score:.1f}/10"


def banner():
    print(f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════╗
║   Agentic AI — ServiceNow Ticket Assignment              ║
║   Live Run Against Real Developer Instance               ║
╚══════════════════════════════════════════════════════════╝{RESET}
  Instance  : {BOLD}{INSTANCE}{RESET}
  User      : {BOLD}{AUTH[0]}{RESET}
  Threshold : Confidence > {BOLD}{config['confidence_threshold']}{RESET} → auto-assign
""")


def print_ticket_block(ticket, predicted_group, conf_score, decision, top_preds):
    auto   = decision["auto_assign"]
    status = f"{GREEN}{BOLD}✅  AUTO-ASSIGNED{RESET}" if auto else f"{YELLOW}{BOLD}🔶  MANUAL TRIAGE{RESET}"

    priority_color = RED if ticket.get("priority","") in ["1 - Critical","1"] else \
                     YELLOW if ticket.get("priority","") in ["2 - High","2"] else RESET

    print(f"\n  {'─'*56}")
    print(f"  {BOLD}Ticket   :{RESET}  {CYAN}{ticket['number']}{RESET}")
    print(f"  {BOLD}Subject  :{RESET}  {ticket['short_description']}")
    print(f"  {BOLD}Category :{RESET}  {ticket.get('category','—')} / {ticket.get('subcategory','—')}")
    print(f"  {BOLD}Priority :{RESET}  {priority_color}{ticket.get('priority','—')}{RESET}")
    print()
    print(f"  {BOLD}Predicted Group  :{RESET}  {CYAN}{predicted_group}{RESET}")
    print(f"  {BOLD}Confidence Score :{RESET}  {confidence_bar(conf_score)}")
    print()
    print(f"  {BOLD}Top 3 Predictions:{RESET}")
    for grp, prob in top_preds:
        arrow = f"{CYAN}→{RESET}" if grp == predicted_group else " "
        print(f"    {arrow}  {grp:<30}  {prob*100:5.1f}%")
    print()
    print(f"  {BOLD}Decision :{RESET}  {status}")
    print(f"  {DIM}  {decision['reason']}{RESET}")


def print_summary(results):
    if not results:
        return
    auto  = sum(1 for r in results if r["auto_assigned"])
    triage = len(results) - auto
    avg_c = sum(r["confidence"] for r in results) / len(results)

    print(f"\n\n  {'═'*56}")
    print(f"  {BOLD}BATCH SUMMARY{RESET}")
    print(f"  {'═'*56}")
    print(f"  Tickets processed : {BOLD}{len(results)}{RESET}")
    print(f"  Auto-assigned     : {GREEN}{BOLD}{auto}{RESET}")
    print(f"  Manual triage     : {YELLOW}{BOLD}{triage}{RESET}")
    print(f"  Avg confidence    : {BOLD}{avg_c:.1f} / 10{RESET}")
    print()
    print(f"  {'Ticket':<14} {'Assigned To':<30} {'Score':>6}  Result")
    print(f"  {'─'*14} {'─'*30} {'─'*6}  {'─'*14}")
    for r in results:
        status = f"{GREEN}Auto-assigned{RESET}" if r["auto_assigned"] else f"{YELLOW}Manual triage{RESET}"
        print(f"  {r['number']:<14} {r['group'][:29]:<30} {r['confidence']:>6.1f}  {status}")
    print()
    print(f"  {DIM}Full audit log → data/audit.log{RESET}")
    print(f"  {DIM}Accuracy report → python scripts/accuracy_report.py{RESET}")
    print()


# ── Status display ────────────────────────────────────────────────────────────
def show_status():
    print(f"\n{BOLD}Current Ticket States in ServiceNow{RESET}\n")
    r = requests.get(
        f"{INSTANCE}/api/now/table/incident",
        auth=AUTH, headers=HEADERS,
        params={
            "sysparm_query": "state=1^ORstate=2",
            "sysparm_limit": 50,
            "sysparm_fields": "number,short_description,assignment_group,state,priority",
            "sysparm_display_value": "true",
        }, timeout=15
    )
    tickets = r.json().get("result", [])
    if not tickets:
        print(f"  {YELLOW}No open tickets found.{RESET}\n")
        return

    assigned   = [t for t in tickets if t.get("assignment_group", {}).get("display_value","").strip()]
    unassigned = [t for t in tickets if not t.get("assignment_group", {}).get("display_value","").strip()]

    print(f"  {GREEN}Assigned   : {len(assigned)}{RESET}")
    print(f"  {YELLOW}Unassigned : {len(unassigned)}{RESET}\n")

    print(f"  {'Number':<14} {'Short Description':<40} {'Assignment Group':<25} State")
    print(f"  {'─'*14} {'─'*40} {'─'*25} {'─'*10}")
    for t in tickets:
        grp   = t.get("assignment_group", {}).get("display_value", "") or f"{YELLOW}(unassigned){RESET}"
        state = t.get("state", {}).get("display_value", "?") if isinstance(t.get("state"), dict) else t.get("state","?")
        desc  = (t.get("short_description","")[:38] + "..") if len(t.get("short_description","")) > 40 else t.get("short_description","")
        print(f"  {t.get('number',''):<14} {desc:<40} {grp:<25} {state}")
    print()


# ── Reset tickets for re-testing ──────────────────────────────────────────────
def reset_tickets():
    print(f"\n{BOLD}{YELLOW}Resetting assigned tickets back to unassigned...{RESET}\n")
    r = requests.get(
        f"{INSTANCE}/api/now/table/incident",
        auth=AUTH, headers=HEADERS,
        params={
            "sysparm_query": "assignment_groupISNOTEMPTY^state=1",
            "sysparm_limit": 50,
            "sysparm_fields": "sys_id,number,short_description",
            "sysparm_display_value": "true",
        }, timeout=15
    )
    tickets = r.json().get("result", [])
    if not tickets:
        print(f"  {YELLOW}No assigned tickets found to reset.{RESET}\n")
        return

    print(f"  Found {len(tickets)} assigned ticket(s) to reset.\n")
    for t in tickets:
        sys_id = t["sys_id"]
        number = t.get("number", "?")
        patch = requests.patch(
            f"{INSTANCE}/api/now/table/incident/{sys_id}",
            auth=AUTH, headers=HEADERS,
            json={"assignment_group": "", "work_notes": "[AI Agent] Reset for re-testing."},
            timeout=15
        )
        if patch.status_code == 200:
            print(f"  {GREEN}✅{RESET}  {number}  cleared assignment group")
        else:
            print(f"  {RED}❌{RESET}  {number}  failed ({patch.status_code})")
        time.sleep(0.2)
    print(f"\n  {GREEN}Done. Tickets are unassigned and ready for re-processing.{RESET}\n")


# ── Core processing loop ──────────────────────────────────────────────────────
def run(watch: bool = False):
    banner()

    # Init agents
    ingestion  = TicketIngestionAgent(config)
    knowledge  = KnowledgeAgent(config)
    historical = HistoricalDataAgent()
    prediction = AssignmentPredictionAgent(config["model"]["path"])
    confidence = ConfidenceScoringEngine()
    decision   = DecisionAgent(config["confidence_threshold"])
    sn_agent   = ServiceNowUpdateAgent(config)
    learning   = LearningAgent(config["database"]["feedback_db"], config["model"]["path"])

    active_groups, deprecated_mapping = knowledge.load_knowledge()

    print(f"  {BOLD}Active Assignment Groups:{RESET}")
    for g in active_groups:
        print(f"    {GREEN}•{RESET}  {g}")
    print()

    poll_count = 0

    while True:
        poll_count += 1
        ts = time.strftime("%H:%M:%S")

        print(f"\n{BOLD}{'═'*58}{RESET}")
        print(f"{BOLD}  [{ts}]  Poll #{poll_count} — Fetching unassigned tickets...{RESET}")
        print(f"{BOLD}{'═'*58}{RESET}")

        tickets = ingestion.fetch_unassigned_tickets(limit=20)

        if not tickets:
            print(f"\n  {GREEN}✅  All tickets are assigned — nothing to process.{RESET}")
            if not watch:
                break
        else:
            print(f"\n  Found {BOLD}{len(tickets)}{RESET} unassigned ticket(s).\n")
            results = []

            for raw_ticket in tickets:
                ticket   = ingestion.normalize_ticket(raw_ticket)
                features = historical.build_features(ticket)

                predicted_group, raw_prob = prediction.predict(features)
                top_preds                 = prediction.predict_top_n(features, n=3)

                # Deprecated group resolution
                if predicted_group in deprecated_mapping:
                    old = predicted_group
                    predicted_group = deprecated_mapping[predicted_group]
                    print(f"  {YELLOW}↪ Deprecated group '{old}' → resolved to '{predicted_group}'{RESET}")

                conf_score = confidence.calculate(raw_prob, ticket, predicted_group, active_groups)
                is_active  = predicted_group in active_groups
                dec        = decision.decide(ticket, predicted_group, conf_score, is_active)

                print_ticket_block(ticket, predicted_group, conf_score, dec, top_preds)

                # Act on the decision
                if dec["auto_assign"]:
                    sn_agent.assign_ticket(ticket["sys_id"], predicted_group)
                else:
                    note = (
                        f"AI Agent: Low-confidence prediction (score={conf_score:.1f}/10). "
                        f"Top suggestion: '{predicted_group}'. Please assign manually."
                    )
                    sn_agent.add_work_note(ticket["sys_id"], note)

                # Store in audit log
                learning.log_decision(
                    ticket_number  = ticket["number"],
                    sys_id         = ticket["sys_id"],
                    predicted_group= predicted_group,
                    confidence     = conf_score,
                    auto_assigned  = dec["auto_assign"],
                    reason         = dec["reason"],
                    top_predictions= top_preds,
                )

                results.append({
                    "number":       ticket["number"],
                    "group":        predicted_group,
                    "confidence":   conf_score,
                    "auto_assigned": dec["auto_assign"],
                })

                time.sleep(0.5)  # Small pause between tickets

            print_summary(results)

        if not watch:
            break

        interval = config["polling"]["interval_seconds"]
        print(f"  {DIM}Sleeping {interval}s... (Ctrl+C to stop){RESET}\n")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}Stopped by user.{RESET}\n")
            break

        # Refresh knowledge on each poll
        knowledge.refresh()
        active_groups, deprecated_mapping = knowledge.load_knowledge()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI ticket assignment agent")
    parser.add_argument("--watch",  action="store_true", help="Keep polling every N seconds")
    parser.add_argument("--reset",  action="store_true", help="Clear all assignment groups for re-testing")
    parser.add_argument("--status", action="store_true", help="Show current ticket states only")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.reset:
        reset_tickets()
    else:
        run(watch=args.watch)
