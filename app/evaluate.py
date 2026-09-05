"""
Basic evaluation for the RAG pipeline.

Runs a fixed set of test questions against the existing vector store and
checks concrete, verifiable properties rather than judging answer quality
by eye: does a relevant question surface sources above threshold, does an
irrelevant question correctly surface none, and do sources come from the
expected document.

This is a regression check, not a full evaluation suite — its purpose is
to catch retrieval breakage after changes to chunking, thresholds, or the
embedding model, run before/after such changes for comparison.
"""

from app.rag import retrieve

TEST_CASES = [
    {
        "question": "What are the key responsibilities?",
        "expect_sources": True,
        "expected_source_file": "sample.pdf",  # adjust to your actual bank doc filename
    },
    {
        "question": "What is the Smart Market Matchmaker?",
        "expect_sources": True,
        "expected_source_file": "Concept project proposal Empowering Rwandan Agri-SMEs and Cooperatives through Digital Innovation for Circular Agriculture in Rwanda.pdf",
    },
    {
        "question": "What is chemistry?",
        "expect_sources": True,
        "expected_source_file": "chemistry revision.pdf",
    },
    {
        "question": "What is the population of Mars?",
        "expect_sources": False,
        "expected_source_file": None,
    },
]


def run_evaluation():
    passed = 0
    failed = 0

    for case in TEST_CASES:
        sources = retrieve(case["question"])
        has_sources = len(sources) > 0

        result = "PASS"
        reason = ""

        if has_sources != case["expect_sources"]:
            result = "FAIL"
            reason = f"expected sources={case['expect_sources']}, got {has_sources}"
        elif has_sources and case["expected_source_file"]:
            actual_files = {s["source_file"] for s in sources}
            if case["expected_source_file"] not in actual_files:
                result = "FAIL"
                reason = f"expected a source from '{case['expected_source_file']}', got {actual_files}"

        print(f"[{result}] \"{case['question']}\"" + (f" — {reason}" if reason else ""))

        if result == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\n{passed}/{passed + failed} test cases passed.")


if __name__ == "__main__":
    run_evaluation()