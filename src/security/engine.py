"""
SECURITY / GOVERNANCE MODULE
==============================
Authentication, RBAC, encryption, audit logging, compliance guardrails.
"""

import os
import json
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import wraps

from src.utils.logger import log


# ── API Key Management ──────────────────────────────────────────────────────

class APIKeyManager:
    """Manage API keys for pipeline access."""

    def __init__(self, keys_file: str = "./configs/api_keys.json"):
        self.keys_file = Path(keys_file)
        self._keys: Dict[str, Dict] = {}
        self._load_keys()

    def _load_keys(self):
        if self.keys_file.exists() and self.keys_file.stat().st_size > 0:
            with open(self.keys_file) as f:
                self._keys = json.load(f)
        # Also load from env
        env_key = os.environ.get("LLMRAG_API_KEY")
        if env_key:
            self._keys[env_key] = {"role": "admin", "name": "env_default"}

    def generate_key(self, name: str, role: str = "user") -> str:
        """Generate a new API key."""
        key = f"llmragv3_{secrets.token_urlsafe(32)}"
        self._keys[key] = {
            "name": name,
            "role": role,
            "created": datetime.now(timezone.utc).isoformat(),
            "active": True,
        }
        self._save_keys()
        log.info(f"API key generated for '{name}' with role '{role}'")
        return key

    def validate_key(self, key: str) -> Optional[Dict]:
        """Validate an API key. Returns key metadata if valid, None if not."""
        data = self._keys.get(key)
        if data and data.get("active", True):
            return data
        return None

    def revoke_key(self, key: str):
        if key in self._keys:
            self._keys[key]["active"] = False
            self._save_keys()

    def _save_keys(self):
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.keys_file, "w") as f:
            json.dump(self._keys, f, indent=2)


# ── Authentication Middleware ───────────────────────────────────────────────

class AuthMiddleware:
    """FastAPI-compatible authentication middleware."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.method = config.get("method", "api_key")
        self.header = config.get("api_key_header", "X-API-Key")
        self.key_manager = APIKeyManager()

    def authenticate(self, request_headers: Dict[str, str]) -> Optional[Dict]:
        """Authenticate a request. Returns user context if valid."""
        if not self.enabled:
            return {"role": "admin", "name": "auth_disabled"}

        if self.method == "api_key":
            key = request_headers.get(self.header, "")
            return self.key_manager.validate_key(key)

        return None


# ── RBAC ────────────────────────────────────────────────────────────────────

class RBACManager:
    """Role-based access control for pipeline resources."""

    PERMISSIONS = {
        "admin": ["read", "write", "delete", "configure", "query", "manage_keys"],
        "user": ["read", "query"],
        "reader": ["read"],
    }

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)

    def check_permission(self, role: str, action: str) -> bool:
        """Check if a role has permission for an action."""
        if not self.enabled:
            return True
        permissions = self.PERMISSIONS.get(role, [])
        return action in permissions


# ── Encryption ──────────────────────────────────────────────────────────────

class EncryptionManager:
    """Handle encryption at rest and in transit."""

    def __init__(self, config: dict):
        self.at_rest = config.get("at_rest", False)
        self.in_transit = config.get("in_transit", True)
        self._key = os.environ.get("LLMRAG_ENCRYPTION_KEY", "").encode() or \
                    secrets.token_bytes(32)

    def hash_content(self, content: str) -> str:
        """Create a secure hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def encrypt_field(self, value: str) -> str:
        """Simple field-level encryption using HMAC."""
        if not self.at_rest:
            return value
        h = hmac.new(self._key, value.encode(), hashlib.sha256)
        return h.hexdigest()


# ── Audit Logger ────────────────────────────────────────────────────────────

class AuditLogger:
    """Log all pipeline operations for compliance."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.log_queries = config.get("log_queries", True)
        self.log_responses = config.get("log_responses", True)
        self.log_file = config.get("log_file", "./logs/audit.log")
        
        if self.enabled:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, details: Dict[str, Any],
                  user: str = "system"):
        """Log an audit event."""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "user": user,
            "details": details,
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_query(self, query: str, user: str = "anonymous"):
        if self.log_queries:
            self.log_event("QUERY", {"query": query[:500]}, user)

    def log_response(self, query: str, answer: str, user: str = "anonymous"):
        if self.log_responses:
            self.log_event("RESPONSE", {
                "query": query[:200],
                "answer_length": len(answer),
                "answer_preview": answer[:200],
            }, user)

    def log_ingestion(self, source: str, doc_count: int, chunk_count: int):
        self.log_event("INGESTION", {
            "source": source,
            "documents": doc_count,
            "chunks": chunk_count,
        })


# ── Compliance Framework ───────────────────────────────────────────────────

class ComplianceManager:
    """Track compliance with regulatory frameworks."""

    FRAMEWORK_CHECKS = {
        "HIPAA": [
            "pii_masking_enabled",
            "encryption_at_rest",
            "audit_logging",
            "access_controls",
            "data_retention_policy",
        ],
        "SOC2": [
            "encryption_in_transit",
            "audit_logging",
            "access_controls",
            "change_management",
        ],
        "GDPR": [
            "pii_masking_enabled",
            "right_to_delete",
            "data_retention_policy",
            "consent_tracking",
        ],
        "FedRAMP": [
            "encryption_at_rest",
            "encryption_in_transit",
            "audit_logging",
            "access_controls",
            "continuous_monitoring",
        ],
    }

    def __init__(self, config: dict):
        self.frameworks = config.get("frameworks", [])
        self.retention_days = config.get("data_retention_days", 90)
        self.right_to_delete = config.get("right_to_delete", True)

    def run_compliance_check(self, pipeline_config: dict) -> Dict[str, Any]:
        """Run compliance checks against configured frameworks."""
        results = {}
        for framework in self.frameworks:
            checks = self.FRAMEWORK_CHECKS.get(framework, [])
            passed = []
            failed = []
            for check in checks:
                if self._evaluate_check(check, pipeline_config):
                    passed.append(check)
                else:
                    failed.append(check)
            results[framework] = {
                "passed": passed,
                "failed": failed,
                "score": len(passed) / len(checks) * 100 if checks else 100,
            }
        return results

    @staticmethod
    def _evaluate_check(check: str, config: dict) -> bool:
        """Evaluate a single compliance check."""
        sec = config.get("security", {})
        ing = config.get("ingestion", {})
        
        checks_map = {
            "pii_masking_enabled": ing.get("preprocessing", {}).get(
                "pii_masking", {}
            ).get("enabled", False),
            "encryption_at_rest": sec.get("encryption", {}).get("at_rest", False),
            "encryption_in_transit": sec.get("encryption", {}).get("in_transit", False),
            "audit_logging": sec.get("audit", {}).get("enabled", False),
            "access_controls": sec.get("authentication", {}).get("enabled", False),
            "data_retention_policy": sec.get("compliance", {}).get(
                "data_retention_days", 0
            ) > 0,
            "right_to_delete": sec.get("compliance", {}).get("right_to_delete", False),
            "consent_tracking": False,  # Requires custom implementation
            "change_management": False,
            "continuous_monitoring": False,
        }
        return checks_map.get(check, False)


# ── Security Engine (Facade) ───────────────────────────────────────────────

class SecurityEngine:
    """Unified security facade for the pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.auth = AuthMiddleware(config.get("authentication", {}))
        self.rbac = RBACManager(config.get("rbac", {}))
        self.encryption = EncryptionManager(config.get("encryption", {}))
        self.audit = AuditLogger(config.get("audit", {}))
        self.compliance = ComplianceManager(config.get("compliance", {}))
        log.info("Security engine initialized")

    def authenticate_request(self, headers: dict) -> Optional[Dict]:
        return self.auth.authenticate(headers)

    def authorize(self, role: str, action: str) -> bool:
        return self.rbac.check_permission(role, action)

    def log_query(self, query: str, user: str = "anonymous"):
        self.audit.log_query(query, user)

    def log_response(self, query: str, answer: str, user: str = "anonymous"):
        self.audit.log_response(query, answer, user)

    def check_compliance(self, full_config: dict) -> Dict:
        return self.compliance.run_compliance_check(full_config)
