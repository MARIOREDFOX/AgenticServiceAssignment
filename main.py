"""
Main Orchestrator
------------------
Bootstraps all agents and runs the continuous polling loop.

Usage:
    python main.py                     # run with config/config.yaml
    python main.py --config <path>     # run with a custom config file
    python main.py --once              # process one batch then exit (useful for testing)
"""

import argparse
import logging
import os
import sys
import time

import yaml

from agents.ingestion_agent import TicketIngestionAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.prediction_agent import AssignmentPredictionAgent
from agents.confidence_engine import ConfidenceScoringEngine
from agents.decision_agent import DecisionAgent
from agents.servicenow_agent import ServiceNowUpdateAgent
from agents.learning_agent import LearningAgent


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(config: dict):
    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file  = config.get("logging", {}).get("log_file", "data/audit.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Single-ticket processing
# ---------------------------------------------------------------------------

def process_ticket(
    raw_ticket: dict,
    ingestion_agent:   TicketIngestionAgent,
    knowledge_agent:   KnowledgeAgent,
    historical_agent:  HistoricalDataAgent,
    prediction_agent:  AssignmentPredictionAgent,
    confidence_engine: ConfidenceScoringEngine,
    decision_agent:    DecisionAgent,
    servicenow_agent:  ServiceNowUpdateAgent,
    learning_agent:    LearningAgent,
    active_groups:     list,
    deprecated_mapping: dict,
):
    logger = logging.getLogger("orchestrator")

    ticket = ingestion_agent.normalize_ticket(raw_ticket)
    sys_id = ticket["sys_id"]
    number = ticket["number"]

    logger.info(f"Processing ticket {number} ({sys_id})")

    # ── Feature engineering ──────────────────────────────────────────────
    features = historical_agent.build_features(ticket)

    # ── Prediction ───────────────────────────────────────────────────────
    predicted_group, raw_probability = prediction_agent.predict(features)
    top_predictions = prediction_agent.predict_top_n(features, n=3)

    # ── Deprecated-group resolution ──────────────────────────────────────
    if predicted_group in deprecated_mapping:
        replacement = deprecated_mapping[predicted_group]
        logger.info(
            f"Group '{predicted_group}' is deprecated → resolved to '{replacement}'"
        )
        predicted_group = replacement

    # ── Confidence scoring ───────────────────────────────────────────────
    confidence = confidence_engine.calculate(
        raw_probability, ticket, predicted_group, active_groups
    )

    # ── Decision ─────────────────────────────────────────────────────────
    is_active = predicted_group in active_groups
    decision = decision_agent.decide(ticket, predicted_group, confidence, is_active)

    # ── Audit log ────────────────────────────────────────────────────────
    learning_agent.log_decision(
        ticket_number=number,
        sys_id=sys_id,
        predicted_group=predicted_group,
        confidence=confidence,
        auto_assigned=decision["auto_assign"],
        reason=decision["reason"],
        top_predictions=top_predictions,
    )

    # ── Act ──────────────────────────────────────────────────────────────
    if decision["auto_assign"]:
        success = servicenow_agent.assign_ticket(sys_id, predicted_group)
        if success:
            logger.info(
                f"✅ {number} → '{predicted_group}' (confidence={confidence})"
            )
        else:
            logger.error(f"❌ Failed to assign {number} in ServiceNow.")
    else:
        # Add a work note so humans know the AI looked at it
        note = (
            f"Low-confidence prediction (score={confidence}). "
            f"Top suggestion: '{predicted_group}'. "
            "Please assign manually."
        )
        servicenow_agent.add_work_note(sys_id, note)
        logger.info(
            f"🔶 {number} flagged for manual triage "
            f"(confidence={confidence}, reason={decision['reason']})"
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic ServiceNow Ticket Assignment")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--once", action="store_true", help="Run one poll cycle then exit")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    setup_logging(config)
    logger = logging.getLogger("orchestrator")
    logger.info("=== Agentic ServiceNow Assignment Service starting ===")

    # ── Initialise agents ─────────────────────────────────────────────────
    ingestion_agent   = TicketIngestionAgent(config)
    knowledge_agent   = KnowledgeAgent(config)
    historical_agent  = HistoricalDataAgent()
    prediction_agent  = AssignmentPredictionAgent(config["model"]["path"])
    confidence_engine = ConfidenceScoringEngine()
    decision_agent    = DecisionAgent(config["confidence_threshold"])
    servicenow_agent  = ServiceNowUpdateAgent(config)
    learning_agent    = LearningAgent(
        config["database"]["feedback_db"],
        config["model"]["path"],
    )

    # ── Load knowledge once at startup ────────────────────────────────────
    active_groups, deprecated_mapping = knowledge_agent.load_knowledge()
    logger.info(f"Active groups: {active_groups}")

    poll_interval   = config["polling"]["interval_seconds"]
    retrain_every_n = 50   # Retrain after every 50 tickets processed
    tickets_since_retrain = 0

    # ── Polling loop ──────────────────────────────────────────────────────
    while True:
        try:
            tickets = ingestion_agent.fetch_unassigned_tickets()

            if not tickets:
                logger.info("No unassigned tickets found. Sleeping…")
            else:
                logger.info(f"Found {len(tickets)} ticket(s) to process.")

                for raw_ticket in tickets:
                    process_ticket(
                        raw_ticket,
                        ingestion_agent,
                        knowledge_agent,
                        historical_agent,
                        prediction_agent,
                        confidence_engine,
                        decision_agent,
                        servicenow_agent,
                        learning_agent,
                        active_groups,
                        deprecated_mapping,
                    )
                    tickets_since_retrain += 1

                # Periodic retraining
                if tickets_since_retrain >= retrain_every_n:
                    logger.info("Triggering periodic model retraining…")
                    if learning_agent.retrain_model():
                        prediction_agent.reload()
                        tickets_since_retrain = 0

                # Refresh knowledge periodically (every cycle)
                knowledge_agent.refresh()
                active_groups, deprecated_mapping = knowledge_agent.load_knowledge()

        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

        if args.once:
            logger.info("--once flag set. Exiting after one cycle.")
            break

        logger.info(f"Sleeping {poll_interval}s before next poll…")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
