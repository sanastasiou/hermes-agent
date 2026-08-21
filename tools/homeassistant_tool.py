"""Home Assistant tool for controlling smart home devices via REST API.

Registers four LLM-callable tools:
- ``ha_list_entities`` -- list/filter entities by domain or area
- ``ha_get_state`` -- get detailed state of a single entity
- ``ha_list_services`` -- list available services (actions) per domain
- ``ha_call_service`` -- call a HA service (turn_on, turn_off, set_temperature, etc.)

Authentication uses a Long-Lived Access Token via ``HASS_TOKEN`` env var.
The HA instance URL is read from ``HASS_URL`` (default: http://homeassistant.local:8123).
"""

import asyncio
import fnmatch
import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Kept for backward compatibility (e.g. test monkeypatching); prefer _get_config().
_HASS_URL: str = ""
_HASS_TOKEN: str = ""

# Operator allow/deny lists for entity filtering (e.g. HASS_ENTITY_DENYLIST).
# Module-level mirrors so tests can monkeypatch; real values are read from the
# active profile env at call time via _get_entity_filters().
_HASS_ENTITY_ALLOWLIST: str = ""
_HASS_ENTITY_DENYLIST: str = ""

# Default cap on entities returned by an unfiltered (or over-broad)
# ha_list_entities call. Real HA installs expose thousands of entities; a bare
# call used to dump all of them into the model context (~135K tokens). Capping
# the *returned* set keeps the payload bounded while the operator can still
# pull everything back by passing a higher ``max``.
DEFAULT_MAX_ENTITIES = 100


def _get_config():
    """Return the active profile's Home Assistant URL and token."""
    return (
        (_HASS_URL or get_secret("HASS_URL", "http://homeassistant.local:8123") or "").rstrip("/"),
        (_HASS_TOKEN or get_secret("HASS_TOKEN", "") or "").strip(),
    )


def _get_entity_filter_config():
    """Return (allowlist, denylist) entity-filter patterns from the active profile.

    Operators can permanently exclude known-dead entities (e.g. a zombie
    ``office_thermostat_*`` cluster left behind by a migration) without a code
    change:

    - ``HASS_ENTITY_DENYLIST`` — comma-separated entity_id prefixes or globs
      (e.g. ``office_thermostat_*``) to always exclude from ha_list_entities.
      Acts as a final veto: it wins even against the allowlist.
    - ``HASS_ENTITY_ALLOWLIST`` — same format; when non-empty it restricts
      results to matching entities (a whitelist).

    Values come from the profile-scoped env (same resolution as
    HASS_URL/HASS_TOKEN), with module-level mirrors for test monkeypatching.
    """
    allow = (
        _HASS_ENTITY_ALLOWLIST
        or (get_secret("HASS_ENTITY_ALLOWLIST", "") or "")
    )
    deny = (
        _HASS_ENTITY_DENYLIST
        or (get_secret("HASS_ENTITY_DENYLIST", "") or "")
    )
    return _parse_entity_filter_patterns(allow), _parse_entity_filter_patterns(deny)


def _parse_entity_filter_patterns(value: str) -> List[str]:
    """Split a comma-separated env value into trimmed, non-empty patterns."""
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _entity_matches(entity_id: str, patterns: List[str]) -> bool:
    """Return True if ``entity_id`` matches any operator pattern.

    Matching is deliberately forgiving so an operator can write
    ``office_thermostat_*`` (the way the device family is usually
    abbreviated) and have it cover ``office_thermostat.0_external_temperature``
    (a real HA entity id whose separator after the device name is a dot):

    - exact match
    - prefix match, treating ``.`` and ``_`` as equivalent at the boundary
      (``office_thermostat.0_x`` matches ``office_thermostat_*``)
    - shell-glob match (``*.valve*``, ``climate.office`` etc.)
    """
    for pat in patterns:
        if not pat:
            continue
        if entity_id == pat:
            return True
        if fnmatch.fnmatch(entity_id, pat):
            return True
        # Family-prefix match. A pattern that ends in ``*`` or a separator
        # (``office_thermostat_*``, ``office_thermostat.``) is a family
        # prefix: it should cover ``office_thermostat.0_x`` by treating the
        # separator after the matched base (``.`` or ``_``) as
        # interchangeable, or the id to end exactly at the base. A bare
        # id like ``climate.office`` without a trailing separator is only
        # an exact/glob match, so it does NOT swallow
        # ``climate.office_zombie``.
        if pat[-1] in "*._":
            base = pat.rstrip("*._")
            if base and entity_id.startswith(base) and (
                len(entity_id) == len(base) or entity_id[len(base)] in "._"
            ):
                return True
    return False


def _apply_operator_filters(
    states: List[Dict[str, Any]],
    allow: List[str],
    deny: List[str],
) -> List[Dict[str, Any]]:
    """Apply the operator allow/deny lists (see ``_get_entity_filter_config``).

    The allowlist narrows to a whitelist; the denylist is a final veto, so an
    entity matching both is excluded. This is the intuitive model for the main
    use case: an operator writes an allowlist of live climate devices and a
    denylist of a specific dead one, and the dead one drops out.
    """
    if not allow and not deny:
        return states
    out = []
    for s in states:
        entity_id = s.get("entity_id", "")
        if allow and not _entity_matches(entity_id, allow):
            continue  # not on the whitelist
        if deny and _entity_matches(entity_id, deny):
            continue  # denylist is a final veto
        out.append(s)
    return out


def _operator_filter_violation(entity_id: str) -> Optional[str]:
    """Return a human-readable reason an entity is excluded by operator config,
    or ``None`` if it may be read.

    Shared by the single-entity read path so a denylisted/whitelist-missed
    entity is refused *before* it is fetched — otherwise an operator who
    denies ``office_thermostat_*`` to keep it out of lists could still read
    the frozen zombie value directly and report it as live (the exact
    28°/22° inconsistency this fix targets). Denylist is a final veto,
    matching ``_apply_operator_filters``.
    """
    allow, deny = _get_entity_filter_config()
    if deny and _entity_matches(entity_id, deny):
        return "excluded by HASS_ENTITY_DENYLIST"
    if allow and not _entity_matches(entity_id, allow):
        return "not in HASS_ENTITY_ALLOWLIST"
    return None


def _normalize_id_list(value: Any) -> set:
    """Accept a list or a comma/space-delimited string of entity_ids; dedupe."""
    if value is None:
        return set()
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        parts = [str(value).strip()]
    return {p for p in parts if p}

# Regex for valid HA entity_id format (e.g. "light.living_room", "sensor.temperature_1")
_ENTITY_ID_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$")

# Regex for valid HA service/domain names (e.g. "light", "turn_on", "shell_command").
# Only lowercase ASCII letters, digits, and underscores — no slashes, dots, or
# other characters that could allow path traversal in URL construction.
# The domain and service are interpolated into /api/services/{domain}/{service},
# so allowing arbitrary strings would enable SSRF via path traversal
# (e.g. domain="../../api/config") or blocked-domain bypass
# (e.g. domain="shell_command/../light").
_SERVICE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Service domains blocked for security -- these allow arbitrary code/command
# execution on the HA host or enable SSRF attacks on the local network.
# HA provides zero service-level access control; all safety must be in our layer.
_BLOCKED_DOMAINS = frozenset({
    "shell_command",    # arbitrary shell commands as root in HA container
    "command_line",     # sensors/switches that execute shell commands
    "python_script",    # sandboxed but can escalate via hass.services.call()
    "pyscript",         # scripting integration with broader access
    "hassio",           # addon control, host shutdown/reboot, stdin to containers
    "rest_command",     # HTTP requests from HA server (SSRF vector)
})


def _get_headers(token: str = "") -> Dict[str, str]:
    """Return authorization headers for HA REST API."""
    if not token:
        _, token = _get_config()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Async helpers (called from sync handlers via run_until_complete)
# ---------------------------------------------------------------------------

def _filter_and_summarize(
    states: list,
    domain: Optional[str] = None,
    area: Optional[str] = None,
    entity_ids: Optional[Any] = None,
    name: Optional[str] = None,
    max_entities: Optional[int] = None,
) -> Dict[str, Any]:
    """Filter raw HA states, then build the compact payload.

    Every filter is applied to the raw state list BEFORE the payload is
    built, so tokens are never spent on entities that would be dropped:

    1. operator allow/deny lists (HASS_ENTITY_ALLOWLIST / HASS_ENTITY_DENYLIST)
    2. caller filters: domain, area, name (substring), explicit entity_ids
    3. the entity cap (``max_entities``, default ``DEFAULT_MAX_ENTITIES``)

    A bare (unfiltered) call therefore returns at most ``DEFAULT_MAX_ENTITIES``
    entities and reports ``truncated: true`` when more matched — instead of
    dumping an entire installation (~thousands of entities) into the model.
    """
    allow, deny = _get_entity_filter_config()
    states = _apply_operator_filters(states, allow, deny)

    id_filter = _normalize_id_list(entity_ids)
    if id_filter:
        # An explicit entity list is the caller's precise intent — it
        # supersedes the broader domain/area/name filters (an entity on the
        # list is returned even if it would not match a co-passed domain).
        states = [s for s in states if s.get("entity_id", "") in id_filter]
    else:
        if domain:
            states = [s for s in states if s.get("entity_id", "").startswith(f"{domain}.")]

        if area:
            area_lower = area.lower()
            states = [
                s for s in states
                if area_lower in (s.get("attributes", {}).get("friendly_name", "") or "").lower()
                or area_lower in (s.get("attributes", {}).get("area", "") or "").lower()
            ]

        if name:
            name_lower = name.lower()
            states = [
                s for s in states
                if name_lower in (s.get("attributes", {}).get("friendly_name", "") or "").lower()
                or name_lower in (s.get("entity_id", "") or "").lower()
            ]

    cap = max_entities if (isinstance(max_entities, int) and not isinstance(max_entities, bool) and max_entities > 0) else DEFAULT_MAX_ENTITIES
    truncated = len(states) > cap

    entities = []
    for s in states[:cap]:
        entities.append({
            "entity_id": s["entity_id"],
            "state": s["state"],
            "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
        })

    result = {"count": len(entities), "entities": entities}
    if truncated:
        result["truncated"] = True
        result["truncated_note"] = (
            f"Showing first {cap} of {len(states)} matching entities. "
            "Refine with domain/area/name filters or an explicit entity_ids "
            "list, or pass a larger max."
        )
    return result


async def _async_list_entities(
    domain: Optional[str] = None,
    area: Optional[str] = None,
    entity_ids: Optional[Any] = None,
    name: Optional[str] = None,
    max_entities: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch entity states from HA; filters are applied before the payload is built."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/states"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_get_headers(hass_token), timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            states = await resp.json()

    return _filter_and_summarize(
        states,
        domain=domain,
        area=area,
        entity_ids=entity_ids,
        name=name,
        max_entities=max_entities,
    )


async def _async_get_state(entity_id: str) -> Dict[str, Any]:
    """Fetch detailed state of a single entity."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/states/{entity_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_get_headers(hass_token), timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()

    return {
        "entity_id": data["entity_id"],
        "state": data["state"],
        "attributes": data.get("attributes", {}),
        "last_changed": data.get("last_changed"),
        "last_updated": data.get("last_updated"),
    }


def _build_service_payload(
    entity_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for a HA service call."""
    payload: Dict[str, Any] = {}
    if data:
        payload.update(data)
    # entity_id parameter takes precedence over data["entity_id"]
    if entity_id:
        payload["entity_id"] = entity_id
    return payload


def _parse_service_response(
    domain: str,
    service: str,
    result: Any,
) -> Dict[str, Any]:
    """Parse HA service call response into a structured result."""
    affected = []
    if isinstance(result, list):
        for s in result:
            affected.append({
                "entity_id": s.get("entity_id", ""),
                "state": s.get("state", ""),
            })

    return {
        "success": True,
        "service": f"{domain}.{service}",
        "affected_entities": affected,
    }


async def _async_call_service(
    domain: str,
    service: str,
    entity_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call a Home Assistant service."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/services/{domain}/{service}"
    payload = _build_service_payload(entity_id, data)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=_get_headers(hass_token),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    return _parse_service_response(domain, service, result)


# ---------------------------------------------------------------------------
# Sync wrappers (handler signature: (args, **kw) -> str)
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from a sync handler."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop -- create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    else:
        return asyncio.run(coro)


def _handle_list_entities(args: dict, **kw) -> str:
    """Handler for ha_list_entities tool."""
    domain = args.get("domain")
    area = args.get("area")
    entity_ids = args.get("entity_ids")
    name = args.get("name")
    max_entities = args.get("max")
    if isinstance(max_entities, str):
        try:
            max_entities = int(max_entities)
        except ValueError:
            return tool_error(f"Invalid 'max' value: {max_entities!r} (expected a positive integer)")
    try:
        result = _run_async(_async_list_entities(
            domain=domain,
            area=area,
            entity_ids=entity_ids,
            name=name,
            max_entities=max_entities,
        ))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_list_entities error: %s", e)
        return tool_error(f"Failed to list entities: {e}")


def _handle_get_state(args: dict, **kw) -> str:
    """Handler for ha_get_state tool."""
    entity_id = args.get("entity_id", "")
    if not entity_id:
        return tool_error("Missing required parameter: entity_id")
    if not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")
    violation = _operator_filter_violation(entity_id)
    if violation:
        return tool_error(
            f"{entity_id} is excluded by operator config ({violation}). "
            "Use ha_list_entities with a filter to discover an alternative "
            "entity, or ask the operator to adjust the allow/deny lists."
        )
    try:
        result = _run_async(_async_get_state(entity_id))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_get_state error: %s", e)
        return tool_error(f"Failed to get state for {entity_id}: {e}")


def _handle_call_service(args: dict, **kw) -> str:
    """Handler for ha_call_service tool."""
    domain = args.get("domain", "")
    service = args.get("service", "")
    if not domain or not service:
        return tool_error("Missing required parameters: domain and service")

    # Validate domain/service format BEFORE the blocklist check — prevents
    # path traversal in /api/services/{domain}/{service} and blocklist bypass
    # via payloads like "shell_command/../light".
    if not _SERVICE_NAME_RE.match(domain):
        return tool_error(f"Invalid domain format: {domain!r}")
    if not _SERVICE_NAME_RE.match(service):
        return tool_error(f"Invalid service format: {service!r}")

    if domain in _BLOCKED_DOMAINS:
        return tool_error(
            f"Service domain '{domain}' is blocked for security. "
            f"Blocked domains: {', '.join(sorted(_BLOCKED_DOMAINS))}"
        )

    entity_id = args.get("entity_id")
    if entity_id and not _ENTITY_ID_RE.match(entity_id):
        return tool_error(f"Invalid entity_id format: {entity_id}")

    data = args.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data) if data.strip() else None
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON string in 'data' parameter: {e}")

    try:
        result = _run_async(_async_call_service(domain, service, entity_id, data))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_call_service error: %s", e)
        return tool_error(f"Failed to call {domain}.{service}: {e}")


# ---------------------------------------------------------------------------
# List services
# ---------------------------------------------------------------------------

async def _async_list_services(domain: Optional[str] = None) -> Dict[str, Any]:
    """Fetch available services from HA and optionally filter by domain."""
    import aiohttp

    hass_url, hass_token = _get_config()
    url = f"{hass_url}/api/services"
    headers = {"Authorization": f"Bearer {hass_token}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            services = await resp.json()

    if domain:
        services = [s for s in services if s.get("domain") == domain]

    # Compact the output for context efficiency
    result = []
    for svc_domain in services:
        d = svc_domain.get("domain", "")
        domain_services = {}
        for svc_name, svc_info in svc_domain.get("services", {}).items():
            svc_entry: Dict[str, Any] = {"description": svc_info.get("description", "")}
            fields = svc_info.get("fields", {})
            if fields:
                svc_entry["fields"] = {
                    k: v.get("description", "") for k, v in fields.items()
                    if isinstance(v, dict)
                }
            domain_services[svc_name] = svc_entry
        result.append({"domain": d, "services": domain_services})

    return {"count": len(result), "domains": result}


def _handle_list_services(args: dict, **kw) -> str:
    """Handler for ha_list_services tool."""
    domain = args.get("domain")
    try:
        result = _run_async(_async_list_services(domain=domain))
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("ha_list_services error: %s", e)
        return tool_error(f"Failed to list services: {e}")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_ha_available() -> bool:
    """Tool is only available when HASS_TOKEN is set."""
    return bool(get_secret("HASS_TOKEN"))


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

HA_LIST_ENTITIES_SCHEMA = {
    "name": "ha_list_entities",
    "description": (
        "List Home Assistant entity states. ALWAYS pass at least one filter "
        "(domain, area, name, or an explicit entity_ids list) — an unfiltered "
        "call returns only the first 100 entities and sets truncated=true, so "
        "a bare call is only useful for a rough overview. Prefer domain "
        "(e.g. 'climate', 'sensor') or area/name (e.g. 'kitchen', 'thermostat') "
        "to keep results small and targeted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Entity domain to filter by (e.g. 'light', 'switch', 'climate', "
                    "'sensor', 'binary_sensor', 'cover', 'fan', 'media_player'). "
                    "Strongly preferred over an unfiltered call."
                ),
            },
            "area": {
                "type": "string",
                "description": (
                    "Area/room name to filter by (e.g. 'living room', 'kitchen'). "
                    "Matches against entity friendly names and area attribute."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Substring to match against entity friendly names and "
                    "entity_ids (e.g. 'thermostat', 'temperature'). "
                    "Useful when you don't know the domain."
                ),
            },
            "entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact list of entity_ids to fetch (e.g. "
                    "['climate.office', 'sensor.office_temp']). Returns just "
                    "these entities, ignoring the domain/area/name filters."
                ),
            },
            "max": {
                "type": "integer",
                "description": (
                    "Maximum number of entities to return (default 100). "
                    "Set higher (e.g. 500) only if you genuinely need a "
                    "larger batch and the query is already well filtered."
                ),
            },
        },
        "required": [],
    },
}

HA_GET_STATE_SCHEMA = {
    "name": "ha_get_state",
    "description": (
        "Get the detailed state of a single Home Assistant entity, including all "
        "attributes (brightness, color, temperature setpoint, sensor readings, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": (
                    "The entity ID to query (e.g. 'light.living_room', "
                    "'climate.thermostat', 'sensor.temperature')."
                ),
            },
        },
        "required": ["entity_id"],
    },
}

HA_LIST_SERVICES_SCHEMA = {
    "name": "ha_list_services",
    "description": (
        "List available Home Assistant services (actions) for device control. "
        "Shows what actions can be performed on each device type and what "
        "parameters they accept. Use this to discover how to control devices "
        "found via ha_list_entities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Filter by domain (e.g. 'light', 'climate', 'switch'). "
                    "Omit to list services for all domains."
                ),
            },
        },
        "required": [],
    },
}

HA_CALL_SERVICE_SCHEMA = {
    "name": "ha_call_service",
    "description": (
        "Call a Home Assistant service to control a device. Use ha_list_services "
        "to discover available services and their parameters for each domain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "Service domain (e.g. 'light', 'switch', 'climate', "
                    "'cover', 'media_player', 'fan', 'scene', 'script')."
                ),
            },
            "service": {
                "type": "string",
                "description": (
                    "Service name (e.g. 'turn_on', 'turn_off', 'toggle', "
                    "'set_temperature', 'set_hvac_mode', 'open_cover', "
                    "'close_cover', 'set_volume_level')."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": (
                    "Target entity ID (e.g. 'light.living_room'). "
                    "Some services (like scene.turn_on) may not need this."
                ),
            },
            "data": {
                "type": "string",
                "description": (
                    "Additional service data as a JSON string. Examples: "
                    '{"brightness": 255, "color_name": "blue"} for lights, '
                    '{"temperature": 22, "hvac_mode": "heat"} for climate, '
                    '{"volume_level": 0.5} for media players.'
                ),
            },
        },
        "required": ["domain", "service"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error

registry.register(
    name="ha_list_entities",
    toolset="homeassistant",
    schema=HA_LIST_ENTITIES_SCHEMA,
    handler=_handle_list_entities,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_get_state",
    toolset="homeassistant",
    schema=HA_GET_STATE_SCHEMA,
    handler=_handle_get_state,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_list_services",
    toolset="homeassistant",
    schema=HA_LIST_SERVICES_SCHEMA,
    handler=_handle_list_services,
    check_fn=_check_ha_available,
    emoji="🏠",
)

registry.register(
    name="ha_call_service",
    toolset="homeassistant",
    schema=HA_CALL_SERVICE_SCHEMA,
    handler=_handle_call_service,
    check_fn=_check_ha_available,
    emoji="🏠",
)
