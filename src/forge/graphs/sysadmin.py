"""
Graphe `sysadmin` — diagnostic système en 3 étapes déterministes, jamais de done:false.

Modèle : research.py (search → fetch top N → synthèse LLM unique)
Ici    : discover → collect(logs) → synthèse LLM unique (diagnostic + proposition)

Principes de sécurité (non négociables) :
- Lecture seule stricte. Aucune commande de mutation n'entre dans l'allowlist.
- Toute commande exécutée vient d'un template fixe ; le seul paramètre variable
  (nom de service/container) doit être validé contre la liste retournée par
  l'étape de découverte — jamais interpolé brut depuis une entrée utilisateur
  ou une sortie LLM.
- Timeout + limite de lignes sur chaque commande.
- Logging de chaque commande réellement exécutée (audit).
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field

# --- Config / constantes -----------------------------------------------

DISCOVERY_TIMEOUT_S = 10
COLLECT_TIMEOUT_S = 15
MAX_LOG_LINES = 200          # par commande de collecte
SYSADMIN_MAX_CHARS = 4000    # plafond dédié, même logique que review/research

# Commandes de découverte : listent les candidats, ne consomment aucun
# paramètre variable -> pas de risque d'injection ici.
DISCOVERY_COMMANDS: list[list[str]] = [
    ["systemctl", "list-units", "--type=service", "--no-pager", "--no-legend"],
    ["podman", "ps", "--format", "{{.Names}}"],
]

# Templates de collecte : {name} est substitué UNIQUEMENT après validation
# contre la liste réelle retournée par la découverte (voir _validate_name).
COLLECT_TEMPLATES: dict[str, list[str]] = {
    "journalctl_unit": ["journalctl", "-u", "{name}", "--no-pager", "-n", str(MAX_LOG_LINES)],
    "podman_logs": ["podman", "logs", "--tail", str(MAX_LOG_LINES), "{name}"],
    "journalctl_kernel": ["journalctl", "-k", "--no-pager", "-n", str(MAX_LOG_LINES)],
}


@dataclass
class SysadminResult:
    discovered_units: list[str] = field(default_factory=list)
    discovered_containers: list[str] = field(default_factory=list)
    collected_logs: dict[str, str] = field(default_factory=dict)
    diagnosis: str = ""
    audit_log: list[str] = field(default_factory=list)


# --- Étape 1 : découverte (lecture seule, aucun paramètre variable) -----

def discover() -> tuple[list[str], list[str]]:
    """Liste les units systemd actives et les containers podman en cours.

    Aucune entrée utilisateur ici : commandes fixes, timeout court.
    """
    units: list[str] = []
    containers: list[str] = []

    proc = _run_fixed(DISCOVERY_COMMANDS[0], DISCOVERY_TIMEOUT_S)
    for line in proc.splitlines():
        parts = line.split()
        if parts:
            units.append(parts[0])  # ex: "forge.service"

    proc = _run_fixed(DISCOVERY_COMMANDS[1], DISCOVERY_TIMEOUT_S)
    containers = [line.strip() for line in proc.splitlines() if line.strip()]

    return units, containers


# --- Étape 2 : collecte ciblée (paramètre validé contre la découverte) --

def collect(name: str, kind: str, known_units: list[str], known_containers: list[str]) -> str:
    """Récupère les logs d'un service/container. `name` DOIT figurer dans
    la liste retournée par discover() pour ce `kind` — sinon on refuse.
    """
    if kind == "journalctl_unit":
        if name not in known_units:
            raise ValueError(f"unit inconnue de la découverte, refusé: {name!r}")
    elif kind == "podman_logs":
        if name not in known_containers:
            raise ValueError(f"container inconnu de la découverte, refusé: {name!r}")
    elif kind == "journalctl_kernel":
        pass  # pas de paramètre variable
    else:
        raise ValueError(f"kind de collecte inconnu: {kind!r}")

    template = COLLECT_TEMPLATES[kind]
    cmd = [part.format(name=name) if "{name}" in part else part for part in template]
    return _run_fixed(cmd, COLLECT_TIMEOUT_S)


# --- Étape 3 : synthèse LLM unique (diagnostic + proposition) -----------

def synthesize(collected_logs: dict[str, str], question: str | None, llm_call) -> str:
    """Un seul appel LLM, jamais de done:false. `llm_call` est injecté
    (même pattern que review.py/research.py) pour rester testable.
    """
    prompt = _build_diagnosis_prompt(collected_logs, question)
    raw = llm_call(prompt)
    # Réutilise text_cleaning (strip_think_blocks, try_unwrap_router_json)
    # comme review.py et research.py, pour éviter la divergence par
    # duplication déjà rencontrée en v3.10.
    from forge.text_cleaning import strip_think_blocks, try_unwrap_router_json

    cleaned = try_unwrap_router_json(strip_think_blocks(raw))
    return cleaned[:SYSADMIN_MAX_CHARS]


def _build_diagnosis_prompt(collected_logs: dict[str, str], question: str | None) -> str:
    sections = "\n\n".join(f"--- {name} ---\n{content}" for name, content in collected_logs.items())
    goal = question or "Diagnostique le problème et propose une solution concrète."
    return (
        "Tu es un assistant sysadmin. Voici des extraits de logs.\n"
        f"{sections}\n\n"
        f"{goal}\n"
        "Ne propose jamais d'exécuter une commande toi-même : "
        "décris le diagnostic et la solution en prose claire."
    )


# --- Orchestration du graphe (appel unique, dispatchable depuis le chat) -

def run(target_hint: str | None, question: str | None, llm_call) -> SysadminResult:
    result = SysadminResult()
    units, containers = discover()
    result.discovered_units = units
    result.discovered_containers = containers
    result.audit_log.append("discover: units=%d containers=%d" % (len(units), len(containers)))

    # Sélection des cibles à collecter : cas simple = target_hint fourni et
    # trouvé dans units ou containers ; sinon collecte kernel uniquement.
    # (Le raffinement de sélection multi-cibles est à affiner en v3.11.)
    if target_hint and target_hint in units:
        result.collected_logs["journalctl:" + target_hint] = collect(
            target_hint, "journalctl_unit", units, containers
        )
        result.audit_log.append(f"collect journalctl_unit target={target_hint}")
    elif target_hint and target_hint in containers:
        result.collected_logs["podman:" + target_hint] = collect(
            target_hint, "podman_logs", units, containers
        )
        result.audit_log.append(f"collect podman_logs target={target_hint}")
    else:
        result.collected_logs["kernel"] = collect("", "journalctl_kernel", units, containers)
        result.audit_log.append("collect journalctl_kernel (fallback, pas de cible identifiée)")

    result.diagnosis = synthesize(result.collected_logs, question, llm_call)
    return result


# --- Exécution bas niveau (allowlist stricte, pas de shell=True) --------

def _run_fixed(cmd: list[str], timeout_s: int) -> str:
    """Exécute une commande dont CHAQUE élément est un littéral fixe ou un
    nom déjà validé contre la découverte. Jamais shell=True, jamais de
    construction de chaîne libre.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"[erreur: commande introuvable — {exc}]"
    except subprocess.TimeoutExpired:
        return "[erreur: timeout dépassé]"

    output = proc.stdout or proc.stderr or ""
    lines = output.splitlines()[:MAX_LOG_LINES]
    return "\n".join(lines)
