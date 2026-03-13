"""
Knowledge Agent (Azure Blob Storage)
--------------------------------------
Loads and caches the knowledge article JSON from Azure Blob Storage.
Acts as a governance layer: provides active assignment groups and
deprecated → replacement mappings.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try importing Azure SDK; gracefully degrade if not installed
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("azure-storage-blob not installed. KnowledgeAgent will use local fallback.")


FALLBACK_KNOWLEDGE = {
    "active_assignment_groups": [
        "Network Support",
        "Application Support",
        "Cloud Operations",
    ],
    "deprecated_mapping": {
        "Legacy Network Team": "Network Support",
        "Old App Team": "Application Support",
    },
}


class KnowledgeAgent:
    """
    Retrieves knowledge articles from Azure Blob Storage and caches them
    in memory. Falls back to a local default if Azure is unavailable.
    """

    def __init__(self, config: dict):
        self.config = config["azure_blob"]
        self._active_groups: Optional[list] = None
        self._deprecated_mapping: Optional[dict] = None

    def load_knowledge(self) -> tuple[list, dict]:
        """
        Returns (active_assignment_groups, deprecated_mapping).
        Uses cached value if already loaded; re-fetches from Blob otherwise.
        """
        if self._active_groups is not None:
            return self._active_groups, self._deprecated_mapping

        content = self._fetch_from_blob()

        self._active_groups = content.get("active_assignment_groups", [])
        self._deprecated_mapping = content.get("deprecated_mapping", {})

        logger.info(
            f"Loaded {len(self._active_groups)} active group(s) and "
            f"{len(self._deprecated_mapping)} deprecated mapping(s)."
        )
        return self._active_groups, self._deprecated_mapping

    def refresh(self):
        """
        Forces a reload from Blob Storage on the next call to load_knowledge().
        """
        self._active_groups = None
        self._deprecated_mapping = None
        logger.info("KnowledgeAgent cache cleared. Will reload on next access.")

    def _fetch_from_blob(self) -> dict:
        if not AZURE_AVAILABLE:
            logger.warning("Using fallback knowledge data (Azure SDK not installed).")
            return FALLBACK_KNOWLEDGE

        connection_string = self.config.get("connection_string", "")
        if not connection_string or connection_string == "AZURE_BLOB_CONNECTION_STRING":
            logger.warning("Azure connection string not configured. Using fallback knowledge.")
            return FALLBACK_KNOWLEDGE

        try:
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service.get_container_client(self.config["container_name"])
            blob_client = container_client.get_blob_client(self.config["blob_name"])

            raw_data = blob_client.download_blob().readall()
            content = json.loads(raw_data)
            logger.info("Successfully loaded knowledge article from Azure Blob Storage.")
            return content

        except Exception as e:
            logger.error(f"Failed to load knowledge from Azure Blob: {e}. Using fallback.")
            return FALLBACK_KNOWLEDGE

    def resolve_deprecated(self, group_name: str) -> str:
        """
        If the given group is deprecated, returns its replacement.
        Otherwise returns the group unchanged.
        """
        _, deprecated_mapping = self.load_knowledge()
        return deprecated_mapping.get(group_name, group_name)

    def is_active(self, group_name: str) -> bool:
        """
        Returns True if the group is in the active list.
        """
        active_groups, _ = self.load_knowledge()
        return group_name in active_groups
