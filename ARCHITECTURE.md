# Forge — Vision architecturale

> Forge n'est pas une intelligence qui utilise des outils.
> C'est un système de capacités qui utilise des intelligences.

Ce document capture la trajectoire architecturale long terme de Forge, distincte
de la roadmap produit (voir CHANGELOG / issues GitHub pour les versions v3.x, v4.x...).
Il sert de référence stable pour ne pas avoir à reconstruire ce raisonnement à
chaque nouvelle itération.

## Principe fondateur

L'approche classique part du LLM et lui ajoute des outils :

> "J'ai un LLM, je vais lui ajouter des outils."

Forge inverse cette relation :

> "J'ai un Kernel IA, et un LLM est l'un des outils dont il dispose."

Le LLM devient un *provider* parmi d'autres — un accélérateur cognitif, comme un
GPU est un accélérateur graphique. Le Kernel ne connaît que des capacités
abstraites, jamais une implémentation particulière.

Forge se positionne ainsi non pas comme un framework IA, mais comme un
**système d'exploitation cognitif** : Linux gère le matériel, Forge gère les
capacités cognitives (mémoire, planification, raisonnement, outils, modèles).

## Deux axes distincts

Un piège courant est de confondre :

- **Roadmap produit** : v3.10, v4.0, v5.0... (ce qui est livré, et quand)
- **Maturité architecturale** : Niveau 1 à 4 ci-dessous (comment le système est construit)

Ces deux axes n'avancent pas au même rythme. Un palier architectural peut
arriver plus tard que prévu dans la numérotation produit sans que ce soit un
problème.

## Paliers de maturité architecturale

### Niveau 1 — Kernel monolithique (état actuel, Forge v3.x)

Un Router central appelle directement Memory / Tools / LLM / Planner dans le
même processus. Simple à déboguer, suffisant tant que la complexité reste
gérable dans une seule couche.

```
Router
 ├── Memory
 ├── Tools
 ├── LLM
 └── Planner
```

### Niveau 2 — Les capacités deviennent des interfaces

Au lieu de `memory.search(...)`, on a `MemoryProvider.search(...)`. Le Router
ne connaît que les interfaces, jamais les implémentations. On peut remplacer
SQLite par Chroma, ou un LLM par un autre, sans rien casser ailleurs.

**Migration sans réécriture** : `Capability` devient une interface qui
enveloppe `Tool` plutôt qu'un renommage global —

```
Capability
    ▲
    │
ToolCapability(tool: Tool)
    ▲
    │
Tool existant
```

Les tests existants continuent de passer. Progressivement, certains
composants (Memory, Planner, LLM, OCR) pourront implémenter directement
`Capability` sans jamais avoir été des "tools".

### Niveau 3 — Scheduler

Introduit seulement quand Forge doit exécuter plusieurs tâches en parallèle
(ex. lire un PDF pendant qu'une recherche web et une indexation mémoire
tournent en même temps). C'est un vrai besoin de coordination, pas une
anticipation.

### Niveau 4 — Event Bus

Introduit seulement à ce stade, une fois qu'un besoin réel de coordination
multi-tâches existe (ex. un événement `PDFAnalysé` déclenchant en parallèle
mémoire / résumé / indexation / notifications).

**Règle générale** : un bus d'événements ne doit jamais être un objectif, mais
la conséquence d'un besoin. Chaque palier n'est construit que lorsqu'il
résout un problème concret — pas "pour un futur hypothétique".

## Composants du Kernel

```
Forge Kernel
├── Capability Registry
├── Cognitive Scheduler
│      └── Arbiter
├── Policy Engine
├── Memory System
└── LLM Providers
```

### Capability Registry

Base de connaissances passive : "qui sait faire quoi". Ne choisit jamais.
Pour une capacité donnée (ex. `Reasoning`), il liste les candidats disponibles
(Qwen 2B, Qwen 9B, Cloud LLM...), chacun avec ses caractéristiques (coût,
latence, qualité estimée).

### Cognitive Scheduler (Arbiter)

Le "cerveau du choix" : consulté par le Router une fois l'intention détectée,
il choisit parmi les candidats proposés par le Registry.

**v1 — déterministe.** Pas d'apprentissage. Entrées : capacité demandée +
contexte système + policy + métriques disponibles. Sortie : candidat choisi.
Objectif : simple, testable, prévisible — à l'image d'un scheduler OS
classique (ex. CFS de Linux), qui vise l'équité et la stabilité plutôt que
"l'intelligence" au sens ML.

**v2 — observation.** Ajout d'un Metrics Collector et d'une Decision History
qui observent chaque décision et son résultat, sans encore modifier le
comportement. Construit seulement une fois le v1 réellement en production —
pas en parallèle "pour être prêt".

**Au-delà — apprentissage éventuel.** Le Scheduler pourrait un jour apprendre
de l'usage réel (ex. "requête Python → Qwen 9B directement" après des
milliers de requêtes observées), sans réentraîner de LLM. C'est un axe de
différenciation potentiel : Forge s'améliore par l'expérience d'orchestration,
pas seulement par l'entraînement du modèle. Mais cela suppose un signal de
succès fiable (satisfaction, correction, escalade...), ce qui reste un
problème ouvert — à ne pas confondre avec la simple observation du v2.

### Policy Engine

Composant transversal, pas un "bonus avancé". Décide ce qui est autorisé ou
préférable, une fois l'intention connue :

- Ai-je le droit d'exécuter cette action ? Dois-je demander confirmation ?
- Puis-je utiliser Internet ? Dois-je privilégier le local ?
- Quel est le budget CPU / énergie disponible ?

Exemples concrets : batterie Steam Deck faible → éviter un modèle 14B local,
préférer le NiPoGi ou désactiver des tâches de fond ; NiPoGi hors ligne →
bascule tout en local avec capacités dégradées. C'est ce qui permet à Forge de
s'adapter à différents contextes matériels (Steam Deck sur batterie, serveur
NiPoGi, futur casque XR) sans changer le reste de l'architecture.

### Chaîne de responsabilité complète

```
Message
   │
   ▼
Router (détecte le besoin)
   │
   ▼
Capability Request
   │
   ▼
Capability Registry (liste les candidats)
   │
   ▼
Cognitive Scheduler / Arbiter
   │
   ├── consulte Policy Engine
   ├── consulte Metrics / History (v2+)
   └── choisit le candidat
   │
   ▼
Exécution
```

## Règle des trois phases

Tout composant du Kernel doit passer par trois phases avant d'être considéré
mature :

1. **Primitive** — existe parce qu'il résout un problème concret
   (ex. Capability Registry)
2. **Observable** — son comportement est mesuré
   (ex. Decision History)
3. **Optimisable** — on cherche à l'améliorer automatiquement
   (ex. Learning Scheduler)

Le piège classique est de sauter directement de 1 à 3 : construire Metrics
Collector, Decision History, Analytics et Learning Loop avant même d'avoir un
système observable en production, puis chercher après coup à leur trouver une
utilité. La progression saine est toujours 1 → 2 → 3, dans cet ordre, chaque
étape justifiée par l'usage réel de la précédente.

## Mémoire vs journal de décisions

Deux besoins différents, deux structures différentes :

- **Mémoire sémantique** (`memory_entries` + `memory_vectors`, schéma
  SQLite-vec existant) : pour retrouver du sens — décisions, TODO, contenu
  indexé par embedding.
- **Journal structuré des décisions du Scheduler** (`scheduler_decisions`,
  table dédiée avec colonnes explicites : `capability`, `selected_provider`,
  `device`, `latency_ms`, `estimated_cost`, `confidence`, `context_json`) :
  pour faire des statistiques et de l'audit à l'échelle.

Un JSON libre dans `metadata` suffit pour un journal exploratoire, mais devient
vite un frein dès qu'on veut agréger ("combien de fois Qwen 9B a été choisi sur
Steam Deck pour du reasoning ?"). D'où la table dédiée, créée quand le besoin
apparaît — pas avant.

## Auditabilité

Conséquence directe de cette architecture : le Kernel devient explicable.

```
Décision #48291
Besoin : Code generation
Candidats : Qwen2B, Qwen9B
Refus Qwen2B : score de confiance insuffisant
Politique active : qualité prioritaire
Contexte : NiPoGi secteur, 32 Go RAM disponibles
```

Ce n'est pas qu'un confort de debug : c'est une propriété nécessaire pour un
assistant auquel on délègue des capacités réelles (SSH, Ansible, fichiers,
domotique...). Pour que cette traçabilité existe, le Scheduler v1 doit produire
un `reason` explicite dès sa première version — pas l'ajouter après coup en v2.

## Ce que cette architecture change (et ne change pas)

| | Approche classique | Approche Forge |
|---|---|---|
| Modèles | Plus gros LLM = meilleur assistant | Bonne orchestration > gros modèle seul |
| Mémoire | Tout mettre dans un contexte géant | Indexer, retrouver, injecter ce qui compte |
| Décisions | Laisser le modèle décider | Le Kernel décide, le modèle exécute |
| Évolution | Mettre de l'IA partout | Primitive → Observable → Optimisable |

Le LLM n'est pas la vérité. Le modèle n'est pas le décideur. L'automatisation
n'est pas prioritaire sur la compréhension. Le Kernel doit être prévisible
d'abord, mesurable ensuite, adaptatif seulement éventuellement.

## Pourquoi cette trajectoire est réaliste

Forge v3 peut rester simple aujourd'hui — Router + Tools — tout en ayant une
trajectoire claire et non destructive vers ce Kernel :

```
Aujourd'hui          Demain
Router               Router
 └── Tools             └── Capability Request
                              └── Cognitive Scheduler
                                    └── Capability
```

Rien de tout cela n'impose de réécrire le projet. C'est exactement le sens du
wrapper `ToolCapability` : la bonne architecture n'est pas celle qui prévoit
tout, c'est celle qui permet d'ajouter la prochaine étape sans détruire la
précédente. Pour un projet solo destiné à vivre longtemps sur ses propres
machines (NiPoGi, Steam Deck, peut-être XR plus tard) plutôt qu'à impressionner
pendant une démo de cinq minutes, c'est le compromis le plus sain.
