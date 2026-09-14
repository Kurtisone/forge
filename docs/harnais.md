# Forge Kernel + Harnais — Architecture d'un système cognitif informatique

> Statut : document de conception. Rattache le projet "Harnais/Host Model/World Model" à l'existant Forge (Kernel `kernel-l2-capability` mergé, agent `sysadmin`, échelle de maturité en paliers) plutôt que de le traiter comme un chantier séparé.

---

## Statut d'implémentation (14/09/2026, PR #75, #79, #80)

**La section 9 est livrée, et branchée. Rien d'autre.** Pas de Host
Model, pas de persistance, pas d'`ActionCapability`, rien de la V2 ni de
la V3.

Le Context Builder a d'abord été construit et testé **en isolation** (PR
#75, `graphs/sysadmin.py` intact, un test l'affirmait en lisant sa
source). Il est depuis branché sur les **deux** chemins de ce graphe (PR
#79 et #80) derrière `SYSADMIN_USE_HARNAIS`, défaut `true` — donc actif
en production, le `.env.local` déployé ne portant pas la variable. Le
test « sysadmin intact » est tombé à ce moment-là, comme annoncé ; ce qui
épingle désormais l'ancien comportement est le drapeau à `False`, pas
l'absence du neuf.

| Section | Fichier | État |
|---|---|---|
| 4 (Fact/Hypothesis/Correlation) | [`src/forge/harnais/facts.py`](../src/forge/harnais/facts.py) | livré |
| 3.2 (Collectors) | [`src/forge/harnais/collectors/`](../src/forge/harnais/collectors/) | livré — 3 collectors |
| 3.5 (World Model) | [`src/forge/kernel/world_model.py`](../src/forge/kernel/world_model.py) | livré, en mémoire |
| 3.8 (Context Builder) | [`src/forge/kernel/context_builder.py`](../src/forge/kernel/context_builder.py) | livré, branché sur les deux chemins de `sysadmin` |
| 3.4 (Host Model), 3.6 (persistance), 3.10 (Action Executor), 7 (V2/V3) | — | pas commencé |

### Ce que le texte ci-dessous a de faux

Le document est laissé **tel qu'il a été écrit**. Les écarts sont
enregistrés ici plutôt que corrigés dans le corps, pour que la
conception d'origine et ce qui a réellement été construit restent
lisibles séparément.

1. **§5, `Collector.collect() -> list[Fact]` ne peut pas produire la
   sortie exigée en §6.** `list[Fact]` n'a qu'une façon de dire « aucun
   fait », et un `Fact` ne peut pas porter un échec — donc « proxy en
   panne » et « machine au repos » arrivent identiques au lecteur, ce
   qui est le bug que ce document existe pour corriger. Le code rend une
   `Observation` : les faits, ou la raison qu'il n'y en ait pas. C'est
   la déviation portante ; voir l'en-tête de
   [`harnais/collector.py`](../src/forge/harnais/collector.py).
2. **§9 annonce busctl comme source CPU/RAM.** busctl répond pour les
   *unités* systemd. Rien dans Forge ne lisait la mémoire ni la charge —
   `cpu_ram` est de la plomberie neuve (deux fichiers `/proc`, zéro
   subprocess), pas un wrapper.
3. **§4, `Fact` n'a aucun champ pour l'entité dont il parle.** Les faits
   par conteneur l'encodent dans `key` (`forge-llm.status`). Tient à
   trois collectors ; voudra un vrai champ quand le Host Model de la V2
   nommera les entités.
4. **§4, `WorldEvent` n'a pas de `domain`, alors que §5 filtre dessus.**
   Champ ajouté, et la dataclass gelée : un enregistrement d'audit
   modifiable après coup n'en est pas un.
5. **§4, `Fact.confidence` est montré comme un champ à valeur par
   défaut.** `Literal` n'impose rien à l'exécution : le code le déclare
   `field(init=False)`, donc `Fact(..., confidence="high")` lève
   `TypeError` au lieu d'être seulement déconseillé.

### Ce que le collector conteneurs a gagné depuis (14/09)

Il lit désormais l'heure de démarrage en plus du nom
(`podman ps --format "{{.Names}}\t{{.StartedAt}}\t{{.Status}}"`, endpoint
déjà autorisé par le proxy read-only), et publie un fait `uptime_s` par
conteneur. Raison : la seule panne qu'on sache récurrente ici est
llama-server qui tombe et que `restart: unless-stopped` relève **en
silence** — le 13/09, deux fois, au milieu de mesures. Un `uptime_s` de
247 secondes en face d'une question sur la dernière heure, c'est cette
panne rendue visible, sans historique à consulter et sans arithmétique
que le modèle puisse rater.

Pas de JSON : `_run_fixed` tronque à `SYSADMIN_MAX_LOG_LINES` (100 en
production) et `--format json` indente une vingtaine de lignes par
conteneur, donc cinq conteneurs suffisent à couper le JSON en plein
objet. Un template rend une ligne par conteneur ; et une sortie qui
touche exactement le plafond est refusée, parce que `running_count`
énoncerait sinon le plafond comme un nombre observé.

### Le trou du MVP, et sa fermeture (14/09)

Le collector de logs ne prend **aucune cible** et n'a aucun emplacement
pour en recevoir une : un `LogsCollector(unit=...)` rouvrirait le chemin
« texte choisi par le routeur → `journalctl --unit=` » que tout le modèle
de sécurité de `graphs/sysadmin.py` ferme.

Il est resté fermé. Ce qui a changé, c'est que **le validateur est
arrivé** : `_collect_node`, inchangé et un nœud en amont, vérifie le nom
contre la découverte du même run, et son résultat est enregistré comme
un `Fact` au lieu d'être collé dans un prompt. Les logs par unité sont
donc rendus par le même code, sous les mêmes marqueurs, que tout le
reste — sans qu'aucun collector n'ait gagné de paramètre de cible.

Les deux chemins passent maintenant par le Context Builder
(`SYSADMIN_USE_HARNAIS`, défaut `true`), mesurés avant branchement par
`bench/context_builder_ab.py`.

Ce paragraphe disait, jusqu'au branchement, que rien n'avait été mesuré
contre le vrai modèle. C'était vrai de la PR #75 et ça ne l'est plus :
`bench/context_builder_ab.py` a posé les deux bras au Qwen3.8-9B en
service avant chaque branchement. Ce que le code garantit — aucun énoncé
sans `Fact` derrière — reste vrai quel que soit le modèle et n'avait pas
besoin d'un appel pour l'être ; ce que le bench a mesuré est l'aval,
c'est-à-dire si ce modèle-ci répond *mieux* depuis un contexte honnête
que depuis un bloc de logs. La phase Observable est donc faite pour
`sysadmin`, et pas pour la V2.

---

## 0. Où ça se branche sur l'existant

Avant de définir quoi que ce soit de nouveau, un constat important : **une bonne partie du Harnais existe déjà, sous un autre nom**.

| Ce que tu décris | Ce qui existe déjà dans Forge |
|---|---|
| Harnais = interface d'observation/action sur la machine | Agent `sysadmin` (journalctl, busctl D-Bus, proxy podman read-only), outils `files`/`test`/`shell` |
| Capacités exposées de façon contrôlée, jamais un accès shell brut | `Capability` / `CapabilityRegistry` / `Policy Engine` (branche mergée, PR #25) |
| Permissions, sandboxing, confirmation utilisateur | Policy Engine à 3 flags (`POLICY_ALLOW_NETWORK`, `POLICY_ALLOW_WORKSPACE_WRITES`, `POLICY_ALLOW_SUBPROCESS`) + garde d'escalade déterministe |
| Kernel = orchestrateur qui ne fait pas confiance au LLM comme source de vérité | Philosophie déjà actée : "le LLM n'est pas la vérité, le modèle n'est pas le décideur" |
| Auto-découverte de l'environnement | Découverte libre des services/containers dans `sysadmin` (pas de liste figée en config) |
| Palier 3 (Scheduler), Palier 4 (Event Bus) | Déjà nommés dans l'échelle de maturité, non implémentés faute de besoin réel jusqu'ici |

Ce qui **n'existe pas encore** et que ce document ajoute réellement :
- un **Host Model** persistant (représentation structurée et durable de la machine, pas une simple collecte à la demande) ;
- un **World Model** temporel (historique d'états, événements, corrélation causale) ;
- un **Context Builder** qui sélectionne quoi injecter au LLM plutôt que de tout balancer ;
- la distinction explicite **fait / hypothèse / corrélation / conclusion / confiance** dans les sorties.

Le reste (Capability, Registry, Policy Engine, tool dispatch) est réutilisé tel quel. Le principe des trois phases (Primitive → Observable → Optimisable) et le refus de sauter une étape s'appliquent aussi à ce chantier.

---

## 1. Concepts, précisément

**Harnais** — la seule couche qui touche la machine réelle. Il ne raisonne jamais. Il transforme des observations brutes (sorties de commandes, fichiers, sockets, D-Bus) en **faits structurés et horodatés**. Un fait de Harnais n'est jamais une interprétation : "RAM utilisée = 14.2 GB à 10:32:04" est un fait ; "Forge consomme trop de RAM" n'en est pas un.

**Host Model** — la représentation *statique/lente* de la machine : ce qu'elle est (matériel, OS, services installés, dépendances entre composants). Il change rarement et se construit par auto-découverte progressive.

**World Model** — la représentation *dynamique* : l'état courant et son historique récent (charge CPU, statut d'un container, événements). Il change en continu et porte la notion de séquence temporelle, donc de corrélation causale *probable*.

**Kernel** — le composant qui décide *quoi* observer, *quoi* injecter au LLM, et *quoi* faire d'une décision du LLM. Il ne connaît la machine qu'à travers le Host/World Model, jamais directement.

**LLM** — un fournisseur de raisonnement parmi d'autres, au même titre qu'une `Capability`. Il ne voit jamais l'environnement brut, seulement le contexte que le Kernel a choisi de lui donner. Il ne peut produire que des *hypothèses*, jamais des *faits*.

Distinction centrale à faire respecter partout dans le code, pas seulement dans le prompt :

```
FAIT        : observé directement par le Harnais, horodaté, source traçable
CORRÉLATION : deux faits rapprochés dans le temps, calculée par le Kernel (pas le LLM)
HYPOTHÈSE   : produite par le LLM à partir de faits/corrélations, jamais persistée comme vérité
CONCLUSION  : hypothèse validée par une observation supplémentaire (redevient un fait)
CONFIANCE   : score qualitatif (faible/moyen/élevé) attaché à toute hypothèse, jamais à un fait
```

---

## 2. Frontières Harnais / Kernel / LLM

Règle simple, à ne jamais transgresser dans le code : **une capability ne raisonne pas, le Kernel ne devine pas, le LLM n'observe pas**.

- Le **Harnais** ne connaît pas l'existence du LLM. Il expose des `Capability` d'observation (`ObservationCapability`) et d'action (`ActionCapability`), point.
- Le **Kernel** ne parle jamais directement à `journalctl`, `podman`, ou au système de fichiers — toujours via une Capability du Registry, exactement comme aujourd'hui pour les outils `files`/`shell`.
- Le **LLM** ne reçoit jamais un objet Host/World Model brut. Il reçoit un texte de contexte déjà synthétisé par le Context Builder (même logique que le prompt de synthèse de `sysadmin` aujourd'hui, mais généralisée).

Concrètement, ça se traduit dans le typage :

```python
class ObservationCapability(Capability):
    """Ne retourne jamais de texte libre : un Fact structuré."""

    def observe(self) -> list[Fact]: ...


class ActionCapability(Capability):
    """Hérite des mêmes contraintes que ToolCapability existant :
    Requirements (network/mutates_workspace/spawns_process),
    dry_run obligatoire si risk_level >= MEDIUM."""

    def act(self, intent: ActionIntent) -> ActionResult: ...
```

---

## 3. Architecture des composants

```
                    ┌─────────────────────────────────────────┐
                    │              FORGE KERNEL                │
                    │                                           │
   ┌────────────┐   │  ┌───────────────┐   ┌─────────────────┐ │
   │  Reasoning  │◄──┼──┤ Context       │◄──┤  World Model    │ │
   │  Loop       │   │  │ Builder       │   │  (state+events) │ │
   └─────┬──────┘   │  └───────┬───────┘   └────────▲────────┘ │
         │           │          │                     │          │
         │           │          │           ┌─────────┴────────┐ │
         ▼           │          │           │   Host Model     │ │
   ┌────────────┐    │          │           │  (structure)     │ │
   │ LLM Gateway │    │          │           └────────▲─────────┘ │
   └────────────┘    │          │                     │          │
                      │          ▼                     │          │
   ┌────────────┐     │  ┌───────────────┐    ┌───────┴────────┐│
   │  Action     │◄────┼──┤ Policy Engine │    │  Event Bus     ││
   │  Executor   │     │  │ (existant)    │    │  (interne)     ││
   └─────┬──────┘      │  └───────────────┘    └───────▲────────┘│
         │              │                                │        │
         │              │  ┌─────────────────────────────┘        │
         │              │  │                                      │
         ▼              │  ▼                                      │
   ┌──────────────────────────────┐                                │
   │   Capability Registry        │ (existant, mergé)              │
   │   (passive, ne choisit rien) │                                │
   └──────────────┬───────────────┘                                │
                    │                                               │
                    └──────────────────┬────────────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
                    │              HARNAIS                     │
                    │  ┌────────────┐  ┌────────────────────┐ │
                    │  │ Collectors │  │  Action Capabilities│ │
                    │  │ (observers)│  │  (redémarrer, lire, │ │
                    │  └─────┬──────┘  │   modifier...)      │ │
                    │        │         └──────────┬──────────┘ │
                    │        ▼                    ▼             │
                    │   ┌─────────────────────────────────┐    │
                    │   │  Sandboxing / permissions        │    │
                    │   │  (Policy Engine appliqué ici)    │    │
                    │   └─────────────────────────────────┘    │
                    └───────────────────┬───────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │            WORLD / HOST                     │
                    │  CPU · RAM · GPU · services · containers ·  │
                    │  journald · D-Bus · filesystem · réseau     │
                    └──────────────────────────────────────────────┘
```

### 3.1 Harnais Core
Point d'entrée unique du monde réel. Ne fait *rien* d'intelligent : dispatch vers des `Collector` (lecture) et des `ActionCapability` (écriture), en s'appuyant sur le Registry et le Policy Engine déjà existants. C'est la même position architecturale que `sysadmin` aujourd'hui, généralisée à toute source d'observation.

### 3.2 Collectors
Un Collector par domaine d'observation (CPU/RAM, GPU, réseau, processus, systemd, containers, filesystem, logs, événements noyau). Chaque Collector :
- sait se déclarer "disponible" ou pas sur la machine courante (auto-découverte) ;
- retourne des `Fact` typés, jamais du texte brut ;
- déclare son coût (latence, intrusivité) pour que le Kernel décide s'il vaut la peine de l'interroger.

Réutilise directement les mécanismes déjà en prod : busctl D-Bus pour systemd, proxy podman read-only pour les containers, `journalctl` pour les logs. Pas de nouvelle plomberie à ce niveau, juste une interface commune (`Collector.collect() -> list[Fact]`) autour de ce qui existe.

### 3.3 Event Bus (interne au Kernel, pas le palier 4 global)
Un bus interne, minimal, qui relie Collectors → World Model. Ce n'est **pas** le "Palier 4" de l'échelle de maturité Forge (qui vise la coordination multi-agents) — c'est une brique bien plus modeste : un simple flux d'événements `FactObserved` / `StateChanged` que le World Model consomme. À garder strictement séparé pour ne pas fusionner deux préoccupations différentes.

### 3.4 Host Model
Store de structure, peu volatil. Répond à "de quoi est faite cette machine et comment ses parties dépendent les unes des autres". Construit par une passe d'auto-découverte au démarrage, puis mis à jour de façon incrémentale par les Collectors quand une structure change (nouveau service, nouveau container).

### 3.5 World Model
Store d'état + historique borné. Répond à "que se passe-t-il maintenant, et qu'est-ce qui s'est passé juste avant". Contient :
- l'état courant par métrique/composant ;
- une fenêtre d'événements récents (taille bornée, purge par ancienneté) ;
- les corrélations calculées (règles simples au départ : proximité temporelle + lien de dépendance dans le Host Model — voir 3.6, pas de ML).

### 3.6 State Store / Temporal-Event Memory
Persistance du World Model. Bon candidat de réutilisation : la table `memory_entries` déjà utilisée pour le journal de décisions du scheduler (cf. `forge-kernel`), avec un nouveau type d'entrée (`world_event`) plutôt qu'un store séparé — cohérent avec la décision déjà actée pour le Cognitive Scheduler.

### 3.7 Tool Registry / Permission System / Policy Engine
**Ne pas recréer.** C'est exactement `CapabilityRegistry` + `Policy Engine` du Kernel mergé (PR #25). Les `ActionCapability` du Harnais s'enregistrent dedans comme n'importe quelle capability aujourd'hui.

### 3.8 Context Builder
Nouveau composant, le plus proche en esprit du prompt de synthèse de `sysadmin`. Prend en entrée une intention (question utilisateur ou déclenchement interne), interroge le World/Host Model, et produit un texte de contexte minimal — jamais l'objet complet. C'est lui qui porte la responsabilité de ne pas noyer le LLM (cf. leçon déjà tirée sur le Palier 1 : ne pas injecter les 500+ unités systemd brutes).

### 3.9 Reasoning Loop / LLM Gateway
Le routeur existant, inchangé dans son fonctionnement. Le LLM Gateway reste un `Capability` parmi d'autres (conforme à la vision "LLM = fournisseur, pas centre").

### 3.10 Action Executor
Exécute une décision validée par la Policy Engine. Reprend le mécanisme dry-run/confirmation déjà esquissé pour les opérations dangereuses, avec rollback quand c'est possible (ex. `systemctl` a un état précédent connu, un fichier modifié a un backup).

### 3.11 Audit / Logging
Chaque Fact, chaque décision, chaque action passe par le même journal — extension du logging d'audit déjà en place pour la garde d'escalade.

---

## 4. Structures de données

```python
@dataclass(frozen=True)
class Fact:
    domain: str  # "cpu", "ram", "gpu", "container", "service", ...
    key: str  # "usage_pct", "status", "temperature_c", ...
    value: Any
    unit: str | None
    observed_at: datetime
    source: str  # nom du Collector, pour traçabilité
    confidence: Literal["observed"] = "observed"  # jamais autre chose pour un Fact


@dataclass(frozen=True)
class Correlation:
    facts: tuple[Fact, ...]
    relation: str  # "temporal_proximity", "dependency_chain", ...
    computed_at: datetime
    strength: Literal["weak", "moderate", "strong"]


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    based_on: tuple[Fact | Correlation, ...]
    confidence: Literal["low", "medium", "high"]
    produced_by: str  # "llm:qwen3.5-9b", jamais confondu avec un Fact
    produced_at: datetime


@dataclass
class HostNode:
    name: str  # "forge-llm"
    kind: str  # "container", "service", "package", "device"
    depends_on: list[str]  # noms d'autres HostNode
    metadata: dict[str, Any]
    discovered_at: datetime
    last_confirmed_at: datetime


@dataclass
class WorldEvent:
    event_type: str  # "service_restart", "threshold_crossed", ...
    payload: dict[str, Any]
    timestamp: datetime
    related_nodes: list[str]  # noms de HostNode concernés
```

Le Host Model est un graphe de `HostNode` (dépendances), le World Model une file bornée de `WorldEvent` + un dictionnaire d'état courant `dict[(domain,key), Fact]`.

---

## 5. Interfaces / API

```python
class Collector(Protocol):
    name: str

    def is_available(self) -> bool: ...
    def collect(self) -> list[Fact]: ...
    def cost_hint(self) -> Literal["cheap", "moderate", "expensive"]: ...


class HostModelStore(Protocol):
    def upsert_node(self, node: HostNode) -> None: ...
    def get_dependencies(self, name: str) -> list[HostNode]: ...
    def snapshot(self) -> list[HostNode]: ...


class WorldModelStore(Protocol):
    def record_fact(self, fact: Fact) -> None: ...
    def record_event(self, event: WorldEvent) -> None: ...
    def recent_events(
        self, window: timedelta, domain: str | None = None
    ) -> list[WorldEvent]: ...
    def current_state(self, domain: str) -> list[Fact]: ...


class ContextBuilder(Protocol):
    def build_for(self, intent: str, budget_tokens: int) -> str: ...


class ActionCapability(Capability):
    risk_level: Literal["low", "medium", "high"]

    def dry_run(self, intent: ActionIntent) -> ActionPreview: ...
    def act(self, intent: ActionIntent) -> ActionResult: ...
    def rollback(self, result: ActionResult) -> bool: ...
```

API HTTP (extension de l'existant `GET /tools`, même style) :

```
GET  /harnais/host          -> snapshot du Host Model
GET  /harnais/world?window=5m&domain=ram
POST /harnais/act           -> {capability, intent, dry_run: bool}
GET  /harnais/collectors    -> liste + disponibilité par machine
```

---

## 6. MVP réaliste

Objectif du MVP : **remplacer le diagnostic confabulé de `sysadmin` par un diagnostic honnête**, sans construire tout l'édifice.

Périmètre MVP :
1. 3 Collectors seulement : CPU/RAM, containers (proxy podman existant), logs (journalctl existant).
2. World Model minimal : dictionnaire d'état courant + fenêtre d'événements en mémoire (pas encore persisté dans `memory_entries`).
3. Pas de Host Model séparé au MVP — les dépendances restent codées en dur comme aujourd'hui dans `sysadmin` (ex. `forge-llm` → Vulkan → GPU). Le graphe générique attend le palier suivant.
4. Context Builder = version généralisée du prompt de synthèse `sysadmin` actuel, avec ajout explicite de la distinction fait/hypothèse/confiance dans le prompt.
5. Aucune nouvelle ActionCapability au MVP : uniquement de l'observation. L'action reste ce qui existe déjà (`sysadmin` propose, ne fait rien).

Ce MVP est directement testable sur le cas réel déjà documenté : le run où `containers=0` (proxy HS) menait à un diagnostic confabulé. Avec le MVP, le Context Builder doit produire "impossible d'observer l'état des containers (proxy indisponible) — aucune hypothèse fiable sur cette base" plutôt qu'un faux diagnostic.

---

## 7. Roadmap V1 → V2 → V3

**V1 — Harnais minimal + honnêteté du diagnostic (MVP ci-dessus)**
Collectors CPU/RAM/containers/logs, World Model en mémoire, Context Builder avec distinction fait/hypothèse. Correction directe des défauts sysadmin (1) et (4) déjà identifiés (cible introuvable non signalée, diagnostic confiant sans base).

**V2 — Host Model + auto-découverte + persistance**
Graphe de `HostNode` construit par découverte progressive (au lieu du codage en dur), persistance du World Model dans `memory_entries` (type `world_event`), Collectors supplémentaires (GPU, réseau, systemd via busctl déjà en place). Corrélations simples (proximité temporelle + edge de dépendance dans le Host Model).

**V3 — Actions contrôlées + rollback**
`ActionCapability` réelles (redémarrer un service, appliquer un correctif de config) avec dry-run et rollback, intégrées à la Policy Engine existante. C'est seulement à ce stade que le Kernel passe de "diagnostique" à "agit", et seulement sur ce qui a un rollback connu.

Le passage V2→V3 est le bon moment pour réévaluer si le Palier 3 (Scheduler) ou Palier 4 (Event Bus, au sens Forge) deviennent nécessaires — pas avant, conformément à la règle des trois phases déjà actée.

---

## 8. Difficultés techniques et risques

- **Sur-ingénierie précoce** : le risque principal identifié dans l'historique Forge (paliers 3/4 non justifiés) s'applique directement ici — le Host Model en graphe complet est tentant mais le MVP n'en a pas besoin.
- **Corrélation ≠ causalité** : le World Model ne doit calculer que des corrélations *déclarées comme telles*, jamais présentées comme causales tant qu'aucune confirmation externe n'existe. Risque de régression vers le comportement confabulé déjà documenté si cette règle n'est pas imposée au niveau du type (`Hypothesis` vs `Fact`), pas seulement au niveau du prompt.
- **Coût du Context Builder** : reproduire l'erreur déjà vue (croire que 500+ unités systemd gonflent le prompt alors que ce n'est pas le cas) — mesurer avant d'optimiser, comme la leçon déjà tirée sur `sysadmin` en v3.12.
- **Chaînage non fiable du modèle 9B** : déjà observé trois fois (web_search, memory:recall, files edit) que ce modèle ne chaîne pas deux appels fiablement. Toute action qui nécessiterait observation→décision→action en plusieurs tours du LLM doit être rendue déterministe côté Kernel, pas déléguée au chaînage du modèle.
- **Fraîcheur du Host Model** : un `HostNode` découvert une fois peut devenir obsolète (container supprimé, service désinstallé) — nécessite une notion de `last_confirmed_at` et une politique de péremption, pas un graphe figé.
- **Rollback pas toujours possible** : certaines actions n'ont pas d'état antérieur clair (ex. purge de logs) — la V3 doit accepter des `ActionCapability` sans rollback tant qu'elles sont classées `risk_level="high"` avec confirmation obligatoire.

---

## 9. Première implémentation concrète (point de départ)

Sur la base du MVP (section 6), premier patch réaliste :

1. `forge/harnais/facts.py` — dataclasses `Fact`, `Hypothesis`, `Correlation` (section 4).
2. `forge/harnais/collectors/{cpu_ram,containers,logs}.py` — wrapping des sources déjà utilisées par `sysadmin` (busctl/proxy podman/journalctl), retournant des `Fact` au lieu de texte brut.
3. `forge/kernel/world_model.py` — implémentation en mémoire de `WorldModelStore` (dict + deque bornée), sans persistance au MVP.
4. `forge/kernel/context_builder.py` — reprend le prompt de synthèse `sysadmin`, ajoute la règle : toute affirmation sans `Fact` à l'appui doit être explicitement marquée hypothèse de confiance faible.
5. Test de non-régression ciblé sur le cas déjà documenté (`containers=0` par proxy HS → le Context Builder doit produire un texte disant explicitement que l'observation a échoué, et le test vérifie l'ABSENCE de toute affirmation factuelle sur l'état des containers dans le contexte généré).

Ce point d'entrée ne casse rien de l'existant : `sysadmin` continue de fonctionner, le nouveau Context Builder est branché en remplacement progressif, testable en isolation avant tout remplacement complet.
