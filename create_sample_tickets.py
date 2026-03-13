"""
create_sample_tickets.py
-------------------------
Creates 15 realistic sample incidents directly in your ServiceNow
developer instance — covering all 3 assignment groups so the AI
has a variety of tickets to process.

Run ONCE before starting the agent:
    python create_sample_tickets.py

Requirements:
    pip install requests pyyaml
"""

import sys
import time
import yaml
import requests
from requests.exceptions import RequestException

# ── Load config ───────────────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

INSTANCE = config["servicenow"]["instance_url"].rstrip("/")
AUTH     = (config["servicenow"]["username"], config["servicenow"]["password"])
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}

# ── Colour output ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
YELLOW = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"

# ── 15 Sample tickets across all 3 groups ─────────────────────────────────────
SAMPLE_TICKETS = [

    # ── Network Support (5 tickets) ──────────────────────────────────────────
    {
        "short_description": "VPN not connecting from home office",
        "description": (
            "I am unable to connect to the corporate VPN from my home network. "
            "The Cisco AnyConnect client times out after 30 seconds with error: "
            "'Connection attempt has timed out. Please verify internet connectivity.' "
            "This started this morning. My internet connection is working fine."
        ),
        "category": "Network",
        "subcategory": "VPN",
        "impact": "2",
        "urgency": "2",
        "priority": "2",
        "caller_id": "admin",
    },
    {
        "short_description": "DNS resolution failing for internal domains",
        "description": (
            "Internal DNS lookups are failing for all *.corp.company.com domains. "
            "nslookup returns 'Non-existent domain' for intranet.corp.company.com. "
            "External websites load fine. Affecting 50+ users across the floor."
        ),
        "category": "Network",
        "subcategory": "DNS",
        "impact": "1",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "Firewall blocking outbound HTTPS after policy change",
        "description": (
            "After this morning's firewall policy update, all outbound traffic on port 443 "
            "is blocked for the dev subnet 10.0.5.0/24. All developer workstations "
            "in Building 2 cannot reach GitHub, npm registry, or any external HTTPS sites."
        ),
        "category": "Network",
        "subcategory": "Firewall",
        "impact": "2",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "Wi-Fi dropping every 10 minutes in Building 3",
        "description": (
            "Multiple users on floors 2 and 3 of Building 3 are experiencing Wi-Fi "
            "disconnections every 10–15 minutes since 9 AM today. Reconnecting works "
            "but is disruptive. Wired connections are unaffected."
        ),
        "category": "Network",
        "subcategory": "Wireless",
        "impact": "2",
        "urgency": "2",
        "priority": "2",
        "caller_id": "admin",
    },
    {
        "short_description": "Network switch port showing as down in server room",
        "description": (
            "Switch port Gi0/12 on core switch SW-CORE-01 is showing as administratively "
            "down. The connected server (SRV-DB-04) is unreachable. This is affecting "
            "the database cluster and causing application timeouts."
        ),
        "category": "Network",
        "subcategory": "Hardware",
        "impact": "1",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },

    # ── Application Support (5 tickets) ──────────────────────────────────────
    {
        "short_description": "Cannot login to HR self-service portal",
        "description": (
            "Getting a 403 Forbidden error when trying to access the HR self-service "
            "portal at https://hr.company.com after my password was reset yesterday. "
            "Cleared browser cache, tried Chrome and Edge — same result. "
            "Other users in my team can login fine."
        ),
        "category": "Software",
        "subcategory": "Access",
        "impact": "3",
        "urgency": "2",
        "priority": "3",
        "caller_id": "admin",
    },
    {
        "short_description": "Finance ERP application crashing on launch for all users",
        "description": (
            "The SAP Finance module crashes immediately on startup for all users in "
            "the Finance department (approx 40 users). Error code: 0xC0000005. "
            "This started after this morning's Windows Update patch. "
            "Month-end close is due today — CRITICAL BUSINESS IMPACT."
        ),
        "category": "Application",
        "subcategory": "Error",
        "impact": "1",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "SSO authentication failing for Office 365",
        "description": (
            "Single sign-on for Office 365 is failing with error AADSTS50126: "
            "Invalid username or password. Affects all users trying to login to "
            "Outlook Web Access and Teams from outside the office. "
            "ADFS logs show authentication requests are reaching the server but failing."
        ),
        "category": "Software",
        "subcategory": "Authentication",
        "impact": "1",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "CRM application running extremely slow since upgrade",
        "description": (
            "The Salesforce CRM integration layer has been extremely slow since last "
            "night's v2.4.1 upgrade. Page loads that used to take 2 seconds now take "
            "45+ seconds. Affects all 120 sales team members. "
            "No errors in logs, just extreme latency."
        ),
        "category": "Application",
        "subcategory": "Performance",
        "impact": "2",
        "urgency": "2",
        "priority": "2",
        "caller_id": "admin",
    },
    {
        "short_description": "Password reset portal not sending verification emails",
        "description": (
            "The self-service password reset portal is not sending verification emails. "
            "Users click 'Send Code' but receive nothing. Checked spam folders — "
            "emails are not arriving at all. 15+ users are currently locked out "
            "and cannot reset their passwords."
        ),
        "category": "Software",
        "subcategory": "Email",
        "impact": "2",
        "urgency": "2",
        "priority": "2",
        "caller_id": "admin",
    },

    # ── Cloud Operations (5 tickets) ─────────────────────────────────────────
    {
        "short_description": "Azure VM prod-web-01 stuck in Stopped state",
        "description": (
            "Production virtual machine prod-web-01 (East US region) is stuck in "
            "'Stopped (deallocated)' state after last night's scheduled maintenance. "
            "Cannot start it via Azure Portal — getting 'Operation could not be "
            "completed' error. The website is currently down for all users."
        ),
        "category": "Cloud",
        "subcategory": "Virtual Machine",
        "impact": "1",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "Kubernetes payment-service pod in CrashLoopBackOff",
        "description": (
            "The payment-service deployment in the prod Kubernetes namespace has all "
            "3 pods stuck in CrashLoopBackOff. Error in logs: "
            "'dial tcp 10.0.0.25:5432: connect: connection refused' "
            "Appears the pod cannot reach the PostgreSQL database. "
            "All payment processing is currently down."
        ),
        "category": "Cloud",
        "subcategory": "Kubernetes",
        "impact": "1",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "Azure Blob Storage container returning 500 errors",
        "description": (
            "The Azure Blob Storage container 'prod-assets' in storage account "
            "prodstorageacct01 is returning HTTP 500 Internal Server Error on "
            "all read and write operations. Our application cannot upload or "
            "retrieve files. Affects all document uploads across the platform."
        ),
        "category": "Cloud",
        "subcategory": "Storage",
        "impact": "2",
        "urgency": "1",
        "priority": "1",
        "caller_id": "admin",
    },
    {
        "short_description": "Docker container deployment failing in CI/CD pipeline",
        "description": (
            "All Docker container builds in the Jenkins CI/CD pipeline are failing "
            "with: 'Error response from daemon: No space left on device.' "
            "Build server disk usage is at 100%. "
            "All deployments to staging and production are blocked."
        ),
        "category": "Cloud",
        "subcategory": "Docker",
        "impact": "2",
        "urgency": "2",
        "priority": "2",
        "caller_id": "admin",
    },
    {
        "short_description": "AWS Lambda function timing out intermittently",
        "description": (
            "The order-processor Lambda function in AWS us-east-1 is timing out "
            "on approximately 30% of invocations. Function timeout is set to 30s "
            "but CloudWatch logs show it hitting the limit. Started after last "
            "week's code deploy. Order processing is unreliable."
        ),
        "category": "Cloud",
        "subcategory": "Serverless",
        "impact": "2",
        "urgency": "2",
        "priority": "2",
        "caller_id": "admin",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def test_connection():
    print(f"\n{BOLD}Testing connection to ServiceNow...{RESET}")
    try:
        r = requests.get(
            f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS,
            params={"sysparm_limit": 1}, timeout=15
        )
        if r.status_code == 200:
            print(f"  {GREEN}✅ Connected to {INSTANCE}{RESET}")
            return True
        elif r.status_code == 401:
            print(f"  {RED}❌ Authentication failed — check username/password in config.yaml{RESET}")
        elif r.status_code == 403:
            print(f"  {RED}❌ Forbidden — ensure user has 'rest_api_explorer' and 'itil' roles{RESET}")
        else:
            print(f"  {RED}❌ Unexpected status {r.status_code}: {r.text[:200]}{RESET}")
    except RequestException as e:
        print(f"  {RED}❌ Connection error: {e}{RESET}")
    return False


def get_existing_unassigned_count():
    r = requests.get(
        f"{INSTANCE}/api/now/table/incident",
        auth=AUTH, headers=HEADERS,
        params={"sysparm_query": "assignment_groupISEMPTY^state=1", "sysparm_limit": 100},
        timeout=15
    )
    return len(r.json().get("result", []))


def create_ticket(ticket_data: dict, index: int, total: int) -> dict | None:
    payload = {
        "short_description": ticket_data["short_description"],
        "description":       ticket_data["description"],
        "category":          ticket_data.get("category", ""),
        "subcategory":       ticket_data.get("subcategory", ""),
        "impact":            ticket_data.get("impact", "3"),
        "urgency":           ticket_data.get("urgency", "3"),
        "priority":          ticket_data.get("priority", "3"),
        "state":             "1",   # New
        # Leave assignment_group empty — the AI will fill this in
    }

    try:
        r = requests.post(
            f"{INSTANCE}/api/now/table/incident",
            auth=AUTH, headers=HEADERS, json=payload, timeout=15
        )
        r.raise_for_status()
        result = r.json()["result"]
        number = result.get("number", "N/A")
        sys_id = result.get("sys_id", "")
        print(f"  [{index:02d}/{total}] {GREEN}✅{RESET}  {number}  —  {ticket_data['short_description'][:60]}")
        return {"number": number, "sys_id": sys_id, "short_description": ticket_data["short_description"]}
    except Exception as e:
        print(f"  [{index:02d}/{total}] {RED}❌{RESET}  FAILED  —  {ticket_data['short_description'][:50]}  ({e})")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   ServiceNow Sample Ticket Creator                       ║
║   Creates 15 real incidents in your developer instance   ║
╚══════════════════════════════════════════════════════════╝{RESET}""")

    print(f"\n  Instance : {BOLD}{INSTANCE}{RESET}")
    print(f"  Username : {BOLD}{AUTH[0]}{RESET}")

    if not test_connection():
        sys.exit(1)

    existing = get_existing_unassigned_count()
    if existing > 0:
        print(f"\n  {YELLOW}⚠️  Found {existing} existing unassigned ticket(s) in the instance.{RESET}")
        ans = input("  Create new tickets anyway? (y/n): ").strip().lower()
        if ans != "y":
            print("  Aborted.")
            sys.exit(0)

    print(f"\n{BOLD}Creating {len(SAMPLE_TICKETS)} sample incidents...{RESET}\n")

    created = []
    for i, ticket in enumerate(SAMPLE_TICKETS, 1):
        result = create_ticket(ticket, i, len(SAMPLE_TICKETS))
        if result:
            created.append(result)
        time.sleep(0.3)   # Be polite to the API

    print(f"""
{BOLD}{'═'*60}{RESET}
{BOLD}  Done!{RESET}
  Created  : {GREEN}{BOLD}{len(created)}{RESET} tickets
  Failed   : {RED}{len(SAMPLE_TICKETS) - len(created)}{RESET} tickets

{BOLD}  Tickets are unassigned and waiting for the AI agent.{RESET}

{BOLD}  Next step — run the agent:{RESET}
    {CYAN}python run_agent.py{RESET}
{'═'*60}
""")


if __name__ == "__main__":
    main()
