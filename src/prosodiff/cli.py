"""Command-line interface for Prosodiff."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Optional

import typer

from prosodiff import __version__
from prosodiff.analysis import compare_takes
from prosodiff.demo import write_demo_wavs
from prosodiff.errors import ProsodiffError
from prosodiff.models import AnalysisSettings
from prosodiff.render import render_comparison
from prosodiff.report import write_json_report


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown",
    help="Make matched-delivery prosody differences explicit and visible.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"prosodiff {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = None,
) -> None:
    """Prosodiff compares 2–4 same-text WAV takes without emotion labels."""


def _execute(
    *,
    wavs: list[Path],
    labels: list[str] | None,
    text: str | None,
    output: Path,
    json_output: Path | None,
    settings: AnalysisSettings,
    synthetic_demo: bool,
) -> tuple[Path, Path]:
    comparison = compare_takes(
        wavs,
        labels=labels,
        text=text,
        settings=settings,
        synthetic_demo=synthetic_demo,
    )
    image_path = render_comparison(comparison, output)
    report_path = write_json_report(
        comparison,
        json_output or output.with_suffix(".json"),
    )
    return image_path, report_path


@app.command()
def compare(
    wavs: Annotated[
        list[Path],
        typer.Argument(
            help="Two to four same-speaker, same-text WAV files in comparison order."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="4:5 report path (.png, .svg, or .pdf)."),
    ] = Path("prosodiff-card.png"),
    label: Annotated[
        Optional[list[str]],
        typer.Option(
            "--label",
            "-l",
            help="Take label; repeat once per WAV in the same order.",
        ),
    ] = None,
    text: Annotated[
        Optional[str],
        typer.Option("--text", help="Matched sentence displayed in the report header."),
    ] = None,
    json_output: Annotated[
        Optional[Path],
        typer.Option(
            "--json-output",
            help="Delta-schema path (default: report path with .json suffix).",
        ),
    ] = None,
    fmin: Annotated[
        float,
        typer.Option("--fmin", min=30.0, max=500.0, help="Minimum pYIN F0 in Hz."),
    ] = 50.0,
    fmax: Annotated[
        float,
        typer.Option("--fmax", min=80.0, max=1_200.0, help="Maximum pYIN F0 in Hz."),
    ] = 600.0,
    voicing_threshold: Annotated[
        float,
        typer.Option(
            "--voicing-threshold",
            min=0.0,
            max=1.0,
            help="pYIN probability threshold used only to emphasize contour confidence.",
        ),
    ] = 0.80,
) -> None:
    """Compare two to four matched deliveries and export a card plus JSON."""

    if fmax <= fmin:
        raise typer.BadParameter("--fmax must be greater than --fmin.")
    settings = AnalysisSettings(
        fmin_hz=fmin,
        fmax_hz=fmax,
        voiced_probability_threshold=voicing_threshold,
    )
    try:
        image_path, report_path = _execute(
            wavs=wavs,
            labels=label,
            text=text,
            output=output,
            json_output=json_output,
            settings=settings,
            synthetic_demo=False,
        )
    except ProsodiffError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.secho(f"Created {image_path}", fg=typer.colors.GREEN)
    typer.echo(f"Wrote   {report_path}")


@app.command()
def demo(
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="4:5 demo report path (.png, .svg, or .pdf)."
        ),
    ] = Path("prosodiff-card.png"),
    json_output: Annotated[
        Optional[Path],
        typer.Option(
            "--json-output",
            help="Demo delta-schema path (default: report path with .json suffix).",
        ),
    ] = None,
) -> None:
    """Generate the complete artifact from deterministic synthetic signals."""

    try:
        with TemporaryDirectory(prefix="prosodiff-demo-") as temporary:
            wavs, labels = write_demo_wavs(Path(temporary))
            image_path, report_path = _execute(
                wavs=wavs,
                labels=labels,
                text="Heute schaffen wir das gemeinsam.",
                output=output,
                json_output=json_output,
                settings=AnalysisSettings(),
                synthetic_demo=True,
            )
    except ProsodiffError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.secho(f"Created {image_path}", fg=typer.colors.GREEN)
    typer.echo(f"Wrote   {report_path}")


@app.command()
def ui(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            min=0,
            max=65_535,
            help="Loopback port. Use 0 to select an available port automatically.",
        ),
    ] = 0,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="Open the local interface in the default browser.",
        ),
    ] = True,
) -> None:
    """Open the local browser interface for matched-delivery comparison."""

    from werkzeug.serving import make_server

    from prosodiff.web import create_app

    flask_app = create_app()
    try:
        server = make_server("127.0.0.1", port, flask_app, threaded=True)
    except OSError as exc:
        typer.secho(
            f"Error: could not start the local interface: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    actual_port = int(server.server_port)
    url = f"http://127.0.0.1:{actual_port}/"
    typer.secho("Prosodiff interface is ready.", fg=typer.colors.GREEN)
    typer.echo(url)
    typer.echo(
        "Press Ctrl+C to stop. Uploaded audio stays local and is deleted after analysis."
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopping Prosodiff.")
    finally:
        server.shutdown()
        server.server_close()
        flask_app.extensions["prosodiff_results"].close()
