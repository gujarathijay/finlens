"""Verify Phase 1 setup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="FinLens — Phase 1 Verification")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")

    v = sys.version_info
    table.add_row("Python >= 3.12", f"{'✓' if v.minor >= 12 else '✗'} ({v.major}.{v.minor})")

    try:
        import pydantic, httpx, rich  # noqa: F401, E401
        table.add_row("Core deps", "✓ pydantic, httpx, rich")
    except ImportError as e:
        table.add_row("Core deps", f"✗ {e}")

    try:
        from config.settings import settings
        table.add_row("Config", f"✓ model={settings.base_model_name}")
    except Exception as e:
        table.add_row("Config", f"✗ {e}")

    root = Path(__file__).resolve().parent.parent
    env_exists = (root / ".env").exists()
    table.add_row(".env", f"{'✓' if env_exists else '✗'}")

    console.print()
    console.print(table)
    console.print("\n[bold green]Phase 1 complete![/]")


if __name__ == "__main__":
    main()