# =============================================================================
# TOURNAMENT AUTONOMOUS CONNECTOR
#
# Usage:
#   python tournament_connect.py -a <HOST_IP> -p <PORT>
#
# The agent will connect to the tournament server and keep playing
# game after game automatically. If the connection drops, it
# reconnects after a short delay.
# =============================================================================

import asyncio
import argparse
import time
import sys
import os
from os.path import basename, splitext, dirname

from loguru import logger

# ---------------------------------------------------------------------------
# CONFIG — change these defaults when you know the tournament details
# ---------------------------------------------------------------------------
DEFAULT_HOST = "localhost"   # <-- set the tournament server IP here
DEFAULT_PORT = 16001         # <-- set the tournament server port here
PLAYER_FILE  = "my_player.py"  # the AI file to use

RECONNECT_DELAY_INITIAL = 3    # seconds before first retry
RECONNECT_DELAY_MAX     = 30   # max seconds between retries
# ---------------------------------------------------------------------------


def connect_once(host: str, port: int) -> None:
    """Connect to the tournament master and play until the connection closes."""
    from game_state_hex import GameStateHex
    from seahorse.player.proxies import LocalPlayerProxy

    folder = dirname(os.path.abspath(PLAYER_FILE))
    if folder not in sys.path:
        sys.path.insert(0, folder)

    player_module = __import__(splitext(basename(PLAYER_FILE))[0], fromlist=[None])
    player = LocalPlayerProxy(
        player_module.MyPlayer("B", name=splitext(basename(PLAYER_FILE))[0]),
        gs=GameStateHex
    )

    master_url = f"http://{host}:{port}"
    logger.info(f"Connecting to tournament master at {master_url} ...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            player.listen(keep_alive=True, master_address=master_url)
        )
    finally:
        loop.close()


def run_tournament(host: str, port: int) -> None:
    """Keep connecting and playing — reconnect automatically on any error."""
    delay = RECONNECT_DELAY_INITIAL
    attempt = 0

    logger.info("=" * 60)
    logger.info("  HEX TOURNAMENT — AUTONOMOUS MODE")
    logger.info(f"  Target: {host}:{port}")
    logger.info(f"  Agent:  {PLAYER_FILE}")
    logger.info("=" * 60)

    while True:
        attempt += 1
        logger.info(f"[Attempt #{attempt}] Starting session ...")

        try:
            connect_once(host, port)
            # If we get here cleanly, the server closed the connection
            logger.info("Session ended cleanly. Waiting for next game ...")
            delay = RECONNECT_DELAY_INITIAL   # reset backoff

        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C).")
            break

        except Exception as exc:
            logger.warning(f"Connection lost or error: {exc}")
            logger.info(f"Reconnecting in {delay}s ...")
            time.sleep(delay)
            delay = min(delay * 2, RECONNECT_DELAY_MAX)   # exponential backoff

        else:
            # Clean exit — wait a moment then reconnect for the next game
            time.sleep(RECONNECT_DELAY_INITIAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous Hex tournament connector — plays game after game."
    )
    parser.add_argument(
        "-a", "--address",
        default=DEFAULT_HOST,
        help=f"Tournament server IP (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Tournament server port (default: {DEFAULT_PORT})"
    )
    args = parser.parse_args()

    run_tournament(host=args.address, port=args.port)
