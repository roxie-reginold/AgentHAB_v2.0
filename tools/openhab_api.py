import os
from typing import List, Dict, Any

import requests


class OpenHABAPI:
    """
    Minimal helper around the openHAB REST API, aligned with the official docs:
    https://www.openhab.org/docs/configuration/rules-dsl.html and the REST API docs.

    Only small, composable helpers are provided so agent code can decide how to use
    the data (e.g. to build context for Rules DSL generation).
    """

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.environ.get("OPENHAB_URL", "http://localhost:8080")).rstrip("/")
        self.token = token or os.environ.get("OPENHAB_API_TOKEN") or os.environ.get("OPENHAB_TOKEN")

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # -------------------------------------------------------------------------
    # Items
    # -------------------------------------------------------------------------

    def list_items(self) -> List[Dict[str, Any]]:
        """Return all items from /rest/items."""
        resp = requests.get(f"{self.base_url}/rest/items", headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_item(self, name: str) -> Dict[str, Any] | None:
        """Return a single item description from /rest/items/{name}, or None if 404."""
        try:
            resp = requests.get(f"{self.base_url}/rest/items/{name}", headers=self._headers(), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def create_item(
        self,
        name: str,
        type: str,
        *,
        label: str | None = None,
        tags: List[str] | None = None,
        group_names: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Create an item via PUT /rest/items/{name}. Fails if item already exists (use update_item to change)."""
        payload: Dict[str, Any] = {"type": type, "name": name}
        if label is not None:
            payload["label"] = label
        if tags is not None:
            payload["tags"] = tags
        if group_names is not None:
            payload["groupNames"] = group_names
        resp = requests.put(
            f"{self.base_url}/rest/items/{name}",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return self.get_item(name) or payload

    def send_command(self, item_name: str, command: str) -> None:
        """
        Send a command to an item via /rest/items/{item}/.

        Equivalent to Rules DSL: MyItem.sendCommand("ON").
        """
        url = f"{self.base_url}/rest/items/{item_name}"
        # Text/plain is the expected content type for commands
        resp = requests.post(url, headers={**self._headers(), "Content-Type": "text/plain"}, data=command, timeout=30)
        resp.raise_for_status()

    def post_update(self, item_name: str, state: str) -> None:
        """
        Post an update to an item via /rest/items/{item}/state.

        Equivalent to Rules DSL: MyItem.postUpdate("42").
        """
        url = f"{self.base_url}/rest/items/{item_name}/state"
        resp = requests.put(url, headers={**self._headers(), "Content-Type": "text/plain"}, data=state, timeout=30)
        resp.raise_for_status()

    # -------------------------------------------------------------------------
    # Rules
    # -------------------------------------------------------------------------

    def list_rules(self) -> List[Dict[str, Any]]:
        """
        Return all rules known to the runtime from /rest/rules.

        This includes both UI-created rules and, depending on configuration,
        rules coming from files.
        """
        resp = requests.get(f"{self.base_url}/rest/rules", headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_rule(self, uid: str) -> Dict[str, Any]:
        """Return a single rule description from /rest/rules/{uid}."""
        resp = requests.get(f"{self.base_url}/rest/rules/{uid}", headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def enable_rule(self, uid: str) -> None:
        """Enable a rule via POST /rest/rules/{uid}/enable."""
        resp = requests.post(
            f"{self.base_url}/rest/rules/{uid}/enable",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()

    def disable_rule(self, uid: str) -> None:
        """Disable a rule via POST /rest/rules/{uid}/disable."""
        resp = requests.post(
            f"{self.base_url}/rest/rules/{uid}/disable",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()

    def run_rule(self, uid: str) -> None:
        """
        Manually trigger a rule via POST /rest/rules/{uid}/runnow.

        Useful for testing generated rules.
        """
        resp = requests.post(
            f"{self.base_url}/rest/rules/{uid}/runnow",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()

