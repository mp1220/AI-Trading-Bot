import os
import time
import requests

_last_post = 0


def discord_embed(title, desc, status="info", fields=None, mention_role_id=None):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    color = {"ok": 0x2ECC71, "warn": 0xF1C40F, "error": 0xE74C3C, "info": 0x3498DB}.get(status, 0x95A5A6)
    content = f"<@&{mention_role_id}>" if mention_role_id else None
    payload = {
        "content": content,
        "embeds": [{
            "title": title,
            "description": desc,
            "color": color,
            "fields": [{"name": k, "value": v, "inline": True} for (k, v) in (fields or [])]
        }]
    }
    global _last_post
    if time.time() - _last_post < 2:
        time.sleep(2)
    requests.post(url, json=payload, timeout=10)
    _last_post = time.time()
