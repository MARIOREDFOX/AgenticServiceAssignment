"""
ServiceNow Update Agent
------------------------
Writes the predicted assignment group back to ServiceNow via REST API.
"""

import logging
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class ServiceNowUpdateAgent:
    """
    Handles PATCH requests to the ServiceNow incident table.
    """

    def __init__(self, config: dict):
        self.base_url = config["servicenow"]["instance_url"].rstrip("/")
        self.auth = (
            config["servicenow"]["username"],
            config["servicenow"]["password"],
        )
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def assign_ticket(self, sys_id: str, assignment_group: str) -> bool:
        """
        Updates the assignment_group field on a ServiceNow incident.

        Returns True on success, False on failure.
        """
        url = f"{self.base_url}/api/now/table/incident/{sys_id}"
        payload = {
            "assignment_group": assignment_group,
            "work_notes": (
                f"[AI Assignment Agent] Automatically assigned to "
                f"'{assignment_group}' based on pattern analysis."
            ),
        }

        try:
            response = requests.patch(
                url,
                auth=self.auth,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            logger.info(
                f"Ticket {sys_id} assigned to '{assignment_group}' successfully."
            )
            return True

        except RequestException as e:
            logger.error(f"Failed to assign ticket {sys_id}: {e}")
            return False

    def add_work_note(self, sys_id: str, note: str) -> bool:
        """
        Adds a work note to a ticket without changing assignment.
        Used for manual triage flagging.
        """
        url = f"{self.base_url}/api/now/table/incident/{sys_id}"
        payload = {"work_notes": f"[AI Assignment Agent] {note}"}

        try:
            response = requests.patch(
                url,
                auth=self.auth,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            logger.info(f"Work note added to ticket {sys_id}.")
            return True

        except RequestException as e:
            logger.error(f"Failed to add work note to {sys_id}: {e}")
            return False

    def fetch_resolved_tickets(self, limit: int = 200) -> list[dict]:
        """
        Fetches recently resolved/closed tickets for the learning loop.
        """
        url = f"{self.base_url}/api/now/table/incident"
        params = {
            "sysparm_query": "state=6^ORstate=7^assignment_groupISNOTEMPTY",
            "sysparm_limit": limit,
            "sysparm_fields": (
                "sys_id,short_description,description,category,"
                "subcategory,business_service,priority,assignment_group"
            ),
            "sysparm_display_value": "true",
        }

        try:
            response = requests.get(
                url, auth=self.auth, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()
            tickets = response.json().get("result", [])
            logger.info(f"Fetched {len(tickets)} resolved ticket(s) for feedback loop.")
            return tickets

        except RequestException as e:
            logger.error(f"Failed to fetch resolved tickets: {e}")
            return []
