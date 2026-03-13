"""
train_from_tickets.py
----------------------
Trains the assignment model using the SAME ticket data defined in
create_sample_tickets.py — no CSV file, no synthetic data, no extra steps.

Uses a Random Forest pipeline (with feature scaling) which reliably
separates Cloud / Network / Application tickets based on keyword signals.

Run once before starting the agent:
    python train_from_tickets.py
"""

import os
import sys
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from agents.historical_data_agent import HistoricalDataAgent
from create_sample_tickets import SAMPLE_TICKETS

# ── Map each ticket to its correct assignment group ───────────────────────────
# First 5 = Network, next 5 = Application, last 5 = Cloud (mirrors SAMPLE_TICKETS order)
ASSIGNMENT_GROUPS = (
    ["Network Support"]     * 5 +
    ["Application Support"] * 5 +
    ["Cloud Operations"]    * 5
)

# ── Augmentation: 10 variations per ticket → 150 total training samples ───────
AUGMENTATIONS = [
    lambda t: t,
    lambda t: {**t, "short_description": t["short_description"] + " - urgent"},
    lambda t: {**t, "description": "URGENT: " + t["description"]},
    lambda t: {**t, "priority": "1 - Critical"},
    lambda t: {**t, "priority": "4 - Low"},
    lambda t: {**t, "short_description": "RE: " + t["short_description"]},
    lambda t: {**t, "description": t["description"] + " Please escalate."},
    lambda t: {**t, "business_service": "Corporate IT"},
    lambda t: {**t, "business_service": "Enterprise Systems"},
    lambda t: {**t, "subcategory": ""},
]

GREEN  = "\033[92m"; CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"


def main():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   Training Model from create_sample_tickets.py data      ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

    agent = HistoricalDataAgent()
    X, y = [], []

    for ticket, group in zip(SAMPLE_TICKETS, ASSIGNMENT_GROUPS):
        for aug in AUGMENTATIONS:
            augmented = aug(ticket)
            features  = agent.build_features(augmented)
            X.append(features)
            y.append(group)

    X = np.array(X)
    y = np.array(y)

    classes, counts = np.unique(y, return_counts=True)

    print(f"  Tickets in create_sample_tickets.py  : {len(SAMPLE_TICKETS)}")
    print(f"  Augmentation multiplier              : x{len(AUGMENTATIONS)}")
    print(f"  Total training samples               : {len(X)}")
    print(f"  Feature vector length                : {X.shape[1]}")
    print()
    print(f"  Class distribution:")
    for cls, cnt in zip(classes, counts):
        print(f"    {cls:<30}  {cnt} samples")
    print()

    # Pipeline: scale + Random Forest (handles keyword features much better
    # than Logistic Regression which assumes linear separability)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
        )),
    ])

    cv_folds = 5
    scores   = cross_val_score(model, X, y, cv=cv_folds)
    print(f"  Cross-validation ({cv_folds}-fold) accuracy: "
          f"{scores.mean():.1%}  (±{scores.std():.1%})")
    print()

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/assignment_model.pkl")

    print(f"  {GREEN}✅  Model saved to models/assignment_model.pkl{RESET}")
    print()
    print(f"  Classes the model knows:")
    for cls in model.classes_:
        print(f"    • {cls}")
    print()
    print(f"  {BOLD}Next step:{RESET}")
    print(f"    python create_sample_tickets.py   ← create tickets in ServiceNow")
    print(f"    python run_agent.py               ← run the agent\n")


if __name__ == "__main__":
    main()
