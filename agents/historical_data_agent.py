"""
Historical Data Agent
----------------------
Converts raw ServiceNow ticket fields into numerical feature vectors
used by the Prediction Agent.

Key design: short_description is scored SEPARATELY and with HIGHER WEIGHT
than description, because:
  - It is always filled in (description can be empty or vague)
  - It captures the engineer's primary intent in a few words
  - Categories are often empty or generic — we cannot rely on them
"""

import csv
import hashlib
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def stable_hash(value: str, mod: int = 100) -> int:
    """Deterministic hash of a string, bucketed into [0, mod)."""
    if not value:
        return 0
    digest = int(hashlib.md5(value.lower().encode()).hexdigest(), 16)
    return digest % mod


def keyword_hits(text: str, keywords: set) -> int:
    """Count distinct keyword matches in text. Capped at 5."""
    t = text.lower()
    return min(sum(1 for kw in keywords if kw in t), 5)


# ── Keyword sets (used in both feature engineering AND confidence scoring) ─────
CLOUD_KEYWORDS = {
    "azure", "aws", "gcp", "cloud", "kubernetes", "k8s", "docker",
    "container", "pod", "helm", "terraform", "blob", "s3", "lambda",
    "ec2", "vm", "virtual machine", "pipeline", "ci/cd", "devops",
    "serverless", "cloudfront", "rds", "eks", "aks", "ingress",
    "namespace", "deployment", "node", "cluster", "registry", "vpc",
    "iam", "bucket", "snapshot", "app service", "cloud run",
    "elasticsearch", "grafana", "cloudwatch", "oom", "crashloopbackoff",
    "imagepullbackoff", "kubectl", "helm chart", "terraform", "bicep",
}

NETWORK_KEYWORDS = {
    "vpn", "dns", "firewall", "network", "switch", "router", "wifi",
    "wireless", "dhcp", "vlan", "ip address", "subnet", "port",
    "bandwidth", "latency", "ping", "traceroute", "bgp", "ospf",
    "mpls", "wan", "lan", "ntp", "proxy", "load balancer", "f5",
    "ipsec", "ssl vpn", "voip", "qos", "mac address", "ethernet",
    "isp", "internet", "routing", "peering", "packet loss", "gateway",
    "access point", "ssid", "wap", "arp", "vlan", "trunk",
}

APPLICATION_KEYWORDS = {
    "login", "password", "sso", "authentication", "permission", "access",
    "browser", "chrome", "firefox", "javascript", "api", "sql", "erp",
    "crm", "portal", "email", "report", "batch", "job", "timeout",
    "session", "pdf", "export", "import", "csv", "mobile", "ios",
    "android", "crash", "exception", "log", "config", "upgrade",
    "migration", "patch", "hotfix", "cache", "search", "user account",
    "application error", "software", "module", "plugin", "saml", "ldap",
    "active directory", "totp", "mfa", "2fa", "oauth",
}


class HistoricalDataAgent:

    PRIORITY_MAP = {
        "1 - Critical": 5, "1": 5,
        "2 - High": 4,     "2": 4,
        "3 - Moderate": 3, "3": 3,
        "4 - Low": 2,      "4": 2,
        "5 - Planning": 1, "5": 1,
        "": 0,
    }

    def build_features(self, ticket: dict) -> list:
        """
        Feature vector — short_description is the PRIMARY signal.

        Index  Description
        ─────  ─────────────────────────────────────────────────────
          0    short_description length
          1    description length
          2    combined text length

          3    hash of category        (fallback structural signal)
          4    hash of subcategory
          5    hash of business_service
          6    numeric priority

          ── Short description keyword scores (WEIGHT x2 via duplication) ──
          7    cloud keywords in SHORT DESC        (0–5)
          8    network keywords in SHORT DESC      (0–5)
          9    application keywords in SHORT DESC  (0–5)
         10    cloud keywords in SHORT DESC        (0–5)  ← duplicate for weight
         11    network keywords in SHORT DESC      (0–5)  ← duplicate for weight
         12    application keywords in SHORT DESC  (0–5)  ← duplicate for weight

          ── Full text keyword scores ────────────────────────────────────
         13    cloud keywords in full text         (0–5)
         14    network keywords in full text       (0–5)
         15    application keywords in full text   (0–5)

          ── Category binary flags ───────────────────────────────────────
         16    category == "Cloud"                 (0/1)
         17    category == "Network"               (0/1)
         18    category in ("Application","Software") (0/1)
        """
        short_desc       = ticket.get("short_description", "") or ""
        description      = ticket.get("description", "")      or ""
        category         = ticket.get("category", "")         or ""
        subcategory      = ticket.get("subcategory", "")      or ""
        business_service = ticket.get("business_service", "") or ""
        priority_str     = ticket.get("priority", "")         or ""

        full_text = f"{short_desc} {description} {category} {subcategory}"

        # Short description keyword scores
        sd_cloud = keyword_hits(short_desc, CLOUD_KEYWORDS)
        sd_net   = keyword_hits(short_desc, NETWORK_KEYWORDS)
        sd_app   = keyword_hits(short_desc, APPLICATION_KEYWORDS)

        # Full text keyword scores
        ft_cloud = keyword_hits(full_text, CLOUD_KEYWORDS)
        ft_net   = keyword_hits(full_text, NETWORK_KEYWORDS)
        ft_app   = keyword_hits(full_text, APPLICATION_KEYWORDS)

        return [
            # Structural
            len(short_desc),
            len(description),
            len(short_desc) + len(description),
            stable_hash(category),
            stable_hash(subcategory),
            stable_hash(business_service),
            self.PRIORITY_MAP.get(priority_str, 0),
            # Short description keyword scores (x2 weight via duplication)
            sd_cloud, sd_net, sd_app,
            sd_cloud, sd_net, sd_app,
            # Full text keyword scores
            ft_cloud, ft_net, ft_app,
            # Category binary flags
            1 if category.lower() == "cloud" else 0,
            1 if category.lower() == "network" else 0,
            1 if category.lower() in ("application", "software") else 0,
        ]

    def load_historical_csv(
        self,
        csv_path: str,
        label_column: str = "assignment_group",
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        X, y = [], []
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    label = row.get(label_column, "").strip()
                    if not label:
                        continue
                    ticket = {
                        "short_description": row.get("short_description", ""),
                        "description":       row.get("description", ""),
                        "category":          row.get("category", ""),
                        "subcategory":       row.get("subcategory", ""),
                        "business_service":  row.get("business_service", ""),
                        "priority":          row.get("priority", ""),
                    }
                    X.append(self.build_features(ticket))
                    y.append(label)
            if not X:
                logger.error("No valid rows found in CSV.")
                return None
            return np.array(X), np.array(y)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
