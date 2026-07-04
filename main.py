"""
main.py
-------
Engineering AI Workflow Assistant -- CLI Entry Point

Usage:
    python main.py --report data/reports/your_report.pdf
    python main.py --report data/reports/your_report.pdf --rebuild-index
    python main.py --report data/reports/your_report.pdf --output outputs/my_report.md

The program will:
    1. Build (or load) the FAISS vector index from the knowledge base.
    2. Run the 4-agent LangGraph pipeline on the provided PDF.
    3. Write the final Markdown report to the outputs/ directory.
"""

import sys
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Create console with explicit UTF-8 encoding
console = Console(highlight=False)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="engineering-ai-workflow",
        description="Agentic AI workflow for engineering report analysis using LangGraph + Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --report data/reports/inspection_report.pdf
  python main.py --report data/reports/inspection_report.pdf --rebuild-index
  python main.py --report data/reports/inspection_report.pdf --output outputs/analysis.md
        """,
    )

    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to the engineering PDF report to analyze.",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        default=False,
        help="Force rebuild the FAISS vector index from the knowledge base.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output Markdown report (default: outputs/report.md).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of knowledge base documents to retrieve (default: 5).",
    )

    return parser.parse_args()


def print_banner() -> None:
    """Print the application banner."""
    banner = Text()
    banner.append("Engineering AI Workflow Assistant\n", style="bold cyan")
    banner.append("Powered by ", style="dim")
    banner.append("LangGraph", style="bold green")
    banner.append(" | ", style="dim")
    banner.append("Gemini", style="bold blue")
    banner.append(" | ", style="dim")
    banner.append("FAISS + Sentence Transformers", style="bold yellow")

    console.print(Panel(banner, expand=False, border_style="cyan"))
    console.print()


def main() -> None:
    """Main entry point for the Engineering AI Workflow Assistant."""
    print_banner()
    args = parse_args()

    # -- Step 0: Load configuration ------------------------------------------
    console.print("[bold]Step 0:[/bold] Loading configuration...", style="cyan")
    try:
        from src.config import settings
        settings.validate_api_key()
        console.print(
            f"  [green][OK][/green] API key loaded | Model: [bold]{settings.llm_model}[/bold]\n"
        )
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)

    # -- Step 1: Validate report path ----------------------------------------
    report_path = Path(args.report)
    console.print(f"[bold]Step 1:[/bold] Validating report: [cyan]{report_path}[/cyan]")
    if not report_path.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] PDF not found: {report_path}\n"
            f"  -> Place your PDF in: [cyan]{settings.reports_dir}/[/cyan]\n"
        )
        sys.exit(1)
    console.print(f"  [green][OK][/green] Report found: {report_path.name}\n")

    # Determine output path
    output_path = Path(args.output) if args.output else settings.outputs_dir / "report.md"

    # -- Step 2: Build / Load Vector Store -----------------------------------
    console.print("[bold]Step 2:[/bold] Preparing knowledge base vector index...")
    try:
        from src.rag.loader import load_knowledge_base
        from src.rag.chunking import chunk_documents
        from src.rag.embeddings import build_embeddings
        from src.rag.vectorstore import get_or_build_vector_store

        embeddings = build_embeddings(settings.embedding_model)

        documents = load_knowledge_base(settings.knowledge_base_dir)
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        vectorstore = get_or_build_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            store_dir=settings.vector_store_dir,
            rebuild=args.rebuild_index,
        )
        console.print("  [green][OK][/green] Vector store ready\n")

    except (FileNotFoundError, ValueError) as e:
        console.print(f"\n[bold red]Knowledge Base Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Vector Store Error:[/bold red] {e}")
        sys.exit(1)

    # -- Step 3: Run LangGraph Workflow --------------------------------------
    console.print("[bold]Step 3:[/bold] Running LangGraph agent pipeline...\n")
    try:
        from src.graph.workflow import run_workflow

        final_state = run_workflow(
            pdf_path=str(report_path),
            vectorstore=vectorstore,
            top_k=args.top_k,
        )

    except FileNotFoundError as e:
        console.print(f"\n[bold red]File Error:[/bold red] {e}")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[bold red]Pipeline Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -- Step 4: Generate and Save Report ------------------------------------
    console.print("[bold]Step 4:[/bold] Generating final Markdown report...")
    try:
        from src.utils.report_writer import generate_markdown_report, save_report

        report_content = generate_markdown_report(
            document_output=final_state["document_output"],
            summary_output=final_state["summary_output"],
            retrieval_output=final_state["retrieval_output"],
            recommendation_output=final_state["recommendation_output"],
        )

        save_report(report_content, output_path)

        console.print(f"\n[bold green]Analysis complete![/bold green]")
        console.print(
            Panel(
                f"[bold]Report saved to:[/bold] [cyan]{output_path}[/cyan]",
                border_style="green",
                expand=False,
            )
        )

    except Exception as e:
        console.print(f"\n[bold red]Report Generation Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
