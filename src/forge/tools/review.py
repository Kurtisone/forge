"""
Dispatchable wrapper around forge.graphs.review.

Forge's UI is a single conversational page, zero tabs -- there is no
dedicated review form and there won't be one. This module is what
lets a plain chat message ("relis ce fichier", "vérifie ce fichier et
ses tests") trigger graphs.review.run() through the normal
router → dispatch path, the same way `files` and `memory` already do
for their own multi-field operations.

Interface: run(content: str) -> str
  content is a JSON string:
    {"file_path": "...", "question": "...", "test_path": "..."}
  question and test_path are optional.

To activate: ENABLED_TOOLS=chat,code,review in .env.local
"""

import json

from forge.graphs.review import run as review_run


def run(content: str) -> str:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return '[error] review content must be JSON: {"file_path": "..."}'

    if not isinstance(payload, dict):
        return '[error] review content must be a JSON object: {"file_path": "..."}'

    file_path = payload.get("file_path")
    if not file_path:
        return "[error] missing 'file_path' in review request"

    question = payload.get("question") or "Que peut-on améliorer ?"
    test_path = payload.get("test_path")

    return review_run(file_path, question=question, test_path=test_path)
