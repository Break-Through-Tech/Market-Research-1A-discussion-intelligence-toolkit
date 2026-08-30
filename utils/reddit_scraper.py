import requests
import time
import json

"""
Reddit thread scraper (Arctic Shift) — usage guide
=============================================

Import fetch_and_save_threads() from this module and call it directly.

    from utils.reddit_scraper import fetch_and_save_threads
    fetch_and_save_threads("news", after="2026-08-01", before="2026-08-11")

For a single day, set after/before as adjacent dates:
    fetch_and_save_threads("news", after="2026-08-29", before="2026-08-30")
    # -> all of Aug 29

Output: one JSON object per line (JSONL), one line per thread
(post + its full nested comment tree). Defaults to a file named
"{subreddit}_{after}_{before}.jsonl" unless you pass out_path=.

This uses Arctic Shift (arctic-shift.photon-reddit.com), a free
community-run mirror of Reddit data, not the official Reddit API.
No signup or API key needed. Two things to know because of that:
  - It's a mirror, so very recent posts (last few hours/days) may
    lag behind live Reddit or not show up yet.
  - No uptime guarantee — it's one person's infra. If it's slow or
    down, that's expected.

There's a built-in delay between comment-tree requests (default
0.5s) to be a good citizen on free infrastructure — this is why
larger pulls take a while. You can lower `delay=` if you're in a
hurry, but please don't drop it too low; this endpoint has no auth
and no rate-limit key, so it's on all of us not to hammer it.

Recommended: between ~0.3 and 0.5
"""

BASE = "https://arctic-shift.photon-reddit.com/api"

def fetch_posts(subreddit, after, before, limit="auto"):
    """Fetch posts from a subreddit within a date range.

    Args:
        subreddit: Name of the subreddit (no 'r/' prefix).
        after: Earliest date to include, e.g. "2026-08-01".
        before: Latest date to include, e.g. "2026-08-30".
        limit: Max results per request, or "auto" for 100-1000.

    Returns:
        List of post dicts matching the query.
    """
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "limit": limit,
        "fields": "id,title,selftext,url,created_utc,score,num_comments,author"
    }
    resp = requests.get(f"{BASE}/posts/search", params=params)
    resp.raise_for_status()
    return resp.json()["data"]

def fetch_comment_tree(post_id):
    """Fetch the full nested comment tree for a post, by its ID."""
    params = {"link_id": f"t3_{post_id}"}
    resp = requests.get(f"{BASE}/comments/tree", params=params)
    resp.raise_for_status()
    return resp.json()["data"]

def fetch_and_save_threads(subreddit, after, before, out_path=None, delay=0.5):
    """Fetch full threads (posts + comment trees) and save them to JSONL.

    Writes one JSON object per line, each containing a post and its
    full nested comment tree. Writes incrementally, so a partial run
    still leaves completed threads saved to disk.

    Args:
        subreddit: Name of the subreddit (no 'r/' prefix).
        after: Earliest post date to include, e.g. "2026-08-01".
        before: Latest post date to include, e.g. "2026-08-30".
        out_path: Output file path. Defaults to
            "{subreddit}_{after}_{before}.jsonl" if not given.
        delay: Seconds to wait between comment-tree requests.
    """
    if out_path is None:
        out_path = f"{subreddit}_{after}_{before}.jsonl"

    posts = fetch_posts(subreddit, after, before)
    print(f"Found {len(posts)} posts, fetching comment trees...")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts, 1):
            comments = fetch_comment_tree(post["id"])
            thread = {"post": post, "comments": comments}
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")
            f.flush() 
            print(f"[{i}/{len(posts)}] saved post {post['id']} ({len(comments)} top-level comments)")
            time.sleep(delay)

    print(f"Done. Saved {len(posts)} threads to {out_path}")