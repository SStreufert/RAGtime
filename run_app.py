#!/usr/bin/env python
"""Wrapper to launch app.py with CLI args (--help, --no-track).

Streamlit starts its server BEFORE running the target script, so argparse
inside app.py cannot exit cleanly. This wrapper parses args first, sets
environment variables, then launches Streamlit programmatically.
"""
import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Swan Lagoon RAG app",
        add_help=False,  # We'll add help manually to avoid conflict with REMAINDER
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable MLflow tracking",
    )
    # Allow passing through additional streamlit args (e.g., --server.port)
    parser.add_argument("streamlit_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    # Set env var for the child process
    if args.no_track:
        os.environ["RAG_DISABLE_TRACKING"] = "1"

    # Build streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    if args.streamlit_args:
        cmd.extend(["--"] + args.streamlit_args)

    # Launch streamlit (replaces this process on Unix; on Windows we wait)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()