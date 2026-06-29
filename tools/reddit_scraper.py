#!/usr/bin/env python3
"""
Reddit scraper — fetch posts + comments for PR agent context.

Usage:
  python3 tools/reddit_scraper.py fetch <url> [--output ./reddit_session.json]
  python3 tools/reddit_scraper.py recheck <session_path> [--output ./reddit_session.json]
  python3 tools/reddit_scraper.py summary <session_path> [--format markdown|json]

Environment:
  REDDIT_CLIENT_ID     — Reddit app client ID (optional)
  REDDIT_CLIENT_SECRET — Reddit app client secret (optional)
  REDDIT_USER_AGENT    — User agent string for PRAW (optional)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REDDIT_SESSION_DIR = Path(".reddit_sessions")


def get_reddit_instance():
    """Create a PRAW Reddit instance from env vars, or return None."""
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get(
        "REDDIT_USER_AGENT",
        "linux:npu-gpu-cpu-pr-agent:v1.0 (by /u/bong-water-water-bong)",
    )
    if not client_id or not client_secret:
        return None
    try:
        import praw
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
    except ImportError:
        return None


def parse_reddit_url(url: str):
    """Extract subreddit, post_id from a Reddit URL."""
    m = re.search(r'/r/(\w+)/comments/(\w+)', url)
    if not m:
        raise ValueError(f"Could not parse Reddit URL: {url}")
    return m.group(1), m.group(2)


def do_fetch(url: str, output: Path | None):
    """Fetch a Reddit post and its comments."""
    raise NotImplementedError("do_fetch not yet implemented")


def do_recheck(session: Path, output: Path | None):
    """Re-fetch a previously saved session for new comments."""
    raise NotImplementedError("do_recheck not yet implemented")


def do_summary(session: Path, fmt: str, output: Path | None):
    """Generate a summary from a session."""
    raise NotImplementedError("do_summary not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Reddit scraper for PR agent context")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a Reddit post + comments")
    fetch_parser.add_argument("url", help="Reddit post URL")
    fetch_parser.add_argument("--output", "-o", type=Path, help="Output path (default: .reddit_sessions/<post_id>.json)")

    # recheck
    recheck_parser = subparsers.add_parser("recheck", help="Re-fetch a previously saved session for new comments")
    recheck_parser.add_argument("session", type=Path, help="Path to existing session JSON")
    recheck_parser.add_argument("--output", "-o", type=Path, help="Output path (default: overwrite session)")

    # summary
    summary_parser = subparsers.add_parser("summary", help="Generate a summary from a session")
    summary_parser.add_argument("session", type=Path, help="Path to session JSON")
    summary_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    summary_parser.add_argument("--output", "-o", type=Path, help="Output path (default: stdout)")

    args = parser.parse_args()

    if args.command == "fetch":
        do_fetch(args.url, args.output)
    elif args.command == "recheck":
        do_recheck(args.session, args.output)
    elif args.command == "summary":
        do_summary(args.session, args.format, args.output)


if __name__ == "__main__":
    main()
