# fix/router-files-write — `files:write` via le routeur

Deux bugs préexistants sur `main`, trouvés pendant les tests de pré-merge
du lot 3 sécurité (donc pas des régressions du lot 3). Pris ensemble, ils
faisaient échouer `files:write` depuis le chat pour à peu près **tout
contenu de fichier réel**.

Base : `1cdc26d`. **542 tests verts** (baseline 514), `ruff check` +
`ruff format --check` OK, appliqué et vérifié par `git am` sur un clone
vierge de `main`.

## Le diagnostic

**1. Le scanner d'objets JSON comptait les accolades sans contexte.**
`_all_json_objects` incrémentait sur `{` et décrémentait sur `}` y compris
à l'intérieur des littéraux chaîne. Or le `content` d'une décision routeur
transporte un payload dont le `content` est du texte de fichier — et du
vrai code a des accolades qui ne s'équilibrent pas toutes seules (fonction
Go/C/Rust coupée, `}` dans un commentaire, dict Python). Résultat : la
profondeur retombait à zéro trop tôt (candidat non parsable) ou jamais
(scan abandonné), l'objet était jeté, et la sortie tombait sur le fallback
texte.

`hello.py` marchait — c'est ce qui a masqué le bug si longtemps — parce
que `print('...')` ne contient aucune accolade.

**2. Le double échappement n'est pas tenu par le modèle.**
Imbriquer un payload JSON dans une chaîne JSON impose un second niveau
d'échappement : `\\n` là où une chaîne normale veut `\n`. Le 9B écrit
`\n`. Le parse externe rend alors du texte interne porteur d'un vrai saut
de ligne, et le `json.loads` de l'outil meurt sur *invalid control
character*.

## Les correctifs

**`fix(router)` — scanner conscient des chaînes.** Le comptage saute les
littéraux chaîne, échappements compris. Corrigé au passage : un `{` jamais
refermé faisait `break` et rendait *tout objet ultérieur* inatteignable
(sa fermeture ne ramenait la profondeur que de 2 à 1, jamais à 0) ; on
reprend maintenant un caractère plus loin.

**`feat(router)` + `fix(router)` — `content` DOIT être un objet pour les
outils à payload JSON.** Première tentative : `content ::= string | object`
pour tous les outils. Insuffisant — les deux branches restaient
atteignables, et contre le prior d'un 9B pour la forme chaîne échappée,
les exemples few-shot ont perdu. Confirmé au premier `files:write` réel,
revenu en chaîne échappée et mort sur les guillemets non échappés de
`import "fmt"`.

Correctif réel : la grammaire conditionne la forme de `content` sur
l'outil, ce que GBNF permet puisque `tool` est figé avant `content`.
`root` se scinde en `payload_call` (content = objet, pour
files/memory/review/sysadmin) et `text_call` (content = chaîne, pour
chat/code et les autres). Une branche sans outil est omise (alternation
vide = insatisfiable).

C'est *pourquoi* la contrainte doit vivre dans la grammaire : dans la
forme chaîne, le corps du fichier exige un double échappement que la
grammaire ne peut pas vérifier — pour `schar`, tout le payload n'est que
des caractères. Dans la forme objet, le corps est une chaîne JSON
ordinaire, donc `schar` (qui exclut `"` nu et les caractères de contrôle)
impose son échappement pendant le sampling. Le défaut cesse d'être
rattrapé et devient impossible.

**`feat(router)` — mécanique du ré-encodage.** Le parseur le
ré-encode en `json.dumps`, donc `RouterDecision.content` reste une chaîne
et **aucun contrat `run(content: str)` ne change** : les outils parsent
toujours du texte JSON, produit par Forge au lieu du modèle. La grammaire
GBNF gagne l'alternative objet (avec valeurs JSON complètes, pour qu'un
champ numérique n'ait pas à être stringifié) : la mauvaise forme devient
inatteignable à l'échantillonnage au lieu d'être rattrapée après coup.
La forme chaîne reste acceptée.

**`feat(router)` — le prompt enseigne la forme objet.** Un petit modèle
imite ses exemples bien plus qu'il ne lit les descriptions : tant qu'ils
montraient une chaîne échappée, c'est ce qu'il produisait. Les quatre
outils à payload JSON (`files`, `memory`, `review`, `sysadmin`) basculent
ensemble — deux formes concurrentes côte à côte dans le même prompt, c'est
la recette d'un routage instable sur un 9B (même leçon que l'ambiguïté
`review`/`files` de la v3.10). L'exemple `hello.py` devient multi-ligne :
il ne validait que le seul cas qui n'a jamais posé problème.

**`fix(tools)` — chargeur JSON tolérant partagé.** La forme chaîne ne
disparaît pas (grammaire désactivée, provider sans GBNF, dérive du
modèle). `forge/tool_payload.loads_payload()` réessaie en `strict=False`,
**délibérément en second** : le succès du parse strict est le signal que
le modèle produit du JSON correct, et basculer en tolérant par défaut
masquerait la prochaine régression d'échappement au lieu de l'exposer. La
récupération journalise un avertissement + un événement. Mutualisé pour
les quatre outils plutôt que corrigé dans `files.py` seul — le dépôt s'est
déjà fait avoir deux fois en corrigeant une copie d'un comportement
partagé (`review` vs `research`, cf. `text_cleaning.py`).

## Hors périmètre, corrigé en passant

`memory.run()` appelait `.get()` directement sur le parse : un payload
JSON valide mais non-objet (`"recall"`, une liste) levait `AttributeError`
hors d'un outil dont le contrat est de retourner ses erreurs en texte.
`files`, `review` et `sysadmin` gardaient déjà ce cas.

## Vérification

Bout-en-bout hors tests, routeur → `files.run()` → fichier sur disque :

| contenu | avant | après |
|---|---|---|
| Go équilibré | objet trouvé mais write KO | écrit |
| JS tronqué (`() => {`) | 0 objet, fallback texte | écrit |
| C avec `}` en trop | 0 objet, fallback texte | écrit |
| dict Python | objet trouvé mais write KO | écrit |
| forme chaîne sous-échappée | *invalid control character* | écrit (avec warning) |
| payload réellement malformé | erreur | erreur (inchangé) |

Les tests posent l'invariant plutôt que l'échappement sur lequel ils
cassaient : **tout exemple du prompt doit parser vers l'outil qu'il
nomme**, et aucun exemple à payload JSON ne doit ré-encoder son payload en
chaîne. Un futur exemple mal formé casse désormais un test au lieu
d'apprendre au modèle à échouer.

## Ce qui reste à valider en usage réel

Rien de tout ceci n'a encore tourné contre le vrai modèle : la partie
« le 9B produit-il spontanément la forme objet sous grammaire ? » ne se
prouve qu'en conditions réelles. Test décisif : demander la création d'un
fichier avec du vrai contenu multi-ligne à accolades (un `.go`, un `.js`),
et vérifier que le fichier atterrit sur disque sans passer par le fallback
texte.
