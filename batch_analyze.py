"""
batch_analyze.py
----------------
Batch analyze all PDF reports in the data/reports/ directory.
Each report gets its own output file in outputs/.

Usage:
    python batch_analyze.py
    python batch_analyze.py --rebuild-index   (force rebuild FAISS index first)
"""

import sys
import io
import argparse
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch analyze all PDFs in data/reports/")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index before running")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.config import settings
    settings.validate_api_key()

    # Find all PDFs
    reports_dir = Path("data/reports")
    pdf_files = sorted(reports_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[Batch] No PDF files found in {reports_dir}/")
        sys.exit(1)

    print("=" * 60)
    print(f"  Batch Analyzer — {len(pdf_files)} report(s) found")
    print("=" * 60)
    for i, f in enumerate(pdf_files, 1):
        size_kb = round(f.stat().st_size / 1024, 1)
        print(f"  [{i}] {f.name} ({size_kb} KB)")
    print()

    # Build vector store once (shared across all reports)
    print("[Batch] Preparing knowledge base (shared for all reports)...")
    from src.rag.loader import load_knowledge_base
    from src.rag.chunking import chunk_documents
    from src.rag.embeddings import build_embeddings
    from src.rag.vectorstore import get_or_build_vector_store

    embeddings = build_embeddings(settings.embedding_model)
    documents = load_knowledge_base(settings.knowledge_base_dir)
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vectorstore = get_or_build_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        store_dir=settings.vector_store_dir,
        rebuild=args.rebuild_index,
    )
    print("[Batch] Vector store ready.\n")

    from src.graph.workflow import run_workflow
    from src.utils.report_writer import generate_markdown_report, save_report

    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(pdf_files)}] Analyzing: {pdf_path.name}")
        print(f"{'='*60}")

        # Output filename mirrors the PDF name
        output_name = pdf_path.stem + "_analysis.md"
        output_path = Path("outputs") / output_name

        try:
            final_state = run_workflow(
                pdf_path=str(pdf_path),
                vectorstore=vectorstore,
                top_k=settings.top_k_results,
            )

            report_content = generate_markdown_report(
                document_output=final_state["document_output"],
                summary_output=final_state["summary_output"],
                retrieval_output=final_state["retrieval_output"],
                recommendation_output=final_state["recommendation_output"],
            )

            save_report(report_content, output_path)
            results.append((pdf_path.name, "SUCCESS", str(output_path)))
            print(f"[Batch] [OK] Report saved: {output_path}")

        except Exception as exc:
            print(f"[Batch] [FAIL] {pdf_path.name}: {exc}")
            results.append((pdf_path.name, "FAILED", str(exc)[:80]))

        # Brief pause between LLM calls to avoid rate limits
        if i < len(pdf_files):
            print(f"\n[Batch] Pausing 5s before next report...")
            time.sleep(5)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Batch Complete — Summary")
    print(f"{'='*60}")
    for name, status, detail in results:
        icon = "[OK]" if status == "SUCCESS" else "[FAIL]"
        print(f"  {icon}  {name}")
        print(f"       -> {detail}")

    success_count = sum(1 for _, s, _ in results if s == "SUCCESS")
    print(f"\n  {success_count}/{len(pdf_files)} report(s) generated successfully.")


if __name__ == "__main__":
    main()
