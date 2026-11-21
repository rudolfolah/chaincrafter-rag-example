# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "trulens-dashboard==2.16.0",
#   "trulens-core==2.16.0",
# ]
# ///
"""View persisted Ollama evaluations: uv run --script eval_trulens.py."""

import argparse
import os
import sys
from importlib.resources import files
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trulens-db",
        type=Path,
        default=Path(".trulens/ollama.sqlite"),
        help="Existing TruLens SQLite database (default: .trulens/ollama.sqlite)",
    )
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    database = args.trulens_db.expanduser().resolve()
    if not database.is_file():
        parser.error(f"TruLens database does not exist: {database}")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    # Keep the native dashboard in legacy mode in the same isolated environment.
    os.environ["TRULENS_OTEL_TRACING"] = "0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"

    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    from trulens.core import TruSession

    dashboard_main = Path(str(files("trulens.dashboard").joinpath("main.py"))).resolve()
    if not dashboard_main.is_file():
        parser.error(f"TruLens dashboard entry point does not exist: {dashboard_main}")

    database_url = URL.create("sqlite", database=str(database))
    session = TruSession(database_engine=create_engine(database_url))
    database_connector = session.connector.db
    database_connector.check_db_revision()
    # Launch directly: run_dashboard waits for "Local URL:", but Streamlit emits
    # "URL:" when an explicit address is configured.
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "--server.headless=True",
            "--server.address=127.0.0.1",
            f"--server.port={args.port}",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark",
            "--theme.primaryColor=#E0735C",
            "--theme.font=sans-serif",
            "--server.fileWatcherType=none",
            "--client.toolbarMode=viewer",
            "--global.disableWidgetStateDuplicationWarning=true",
            str(dashboard_main),
            "--",
            "--database-prefix",
            database_connector.table_prefix,
            "--database-url",
            database_connector.engine.url.render_as_string(hide_password=False),
        ],
    )


if __name__ == "__main__":
    main()
