import argparse
import os
from huggingface_hub import upload_folder

def main():
    parser = argparse.ArgumentParser(description="Upload project to Hugging Face Hub cleanly without virtual environment files")
    parser.add_argument("--repo", required=True, help="Hugging Face repo ID (e.g., LordofMonarchs/intelligent-candidate-ranking-system)")
    parser.add_argument("--type", default="space", choices=["space", "model", "dataset"], help="Repository type")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ignore_patterns = [
        ".venv/*",
        ".venv/**",
        "venv/*",
        "venv/**",
        ".git/*",
        ".git/**",
        ".vscode/*",
        ".vscode/**",
        "__pycache__/*",
        "**/__pycache__/*",
        "**/*.py[cod]",
        "candidates.jsonl", # 487MB raw candidates file
        "*.csv",
        "logs/*",
        "logs/**",
        "*.log",
        "reasoning_trace.jsonl",
        "scratch/*",
        "scratch/**",
        "diagnostics/*",
        "diagnostics/**",
        "_tmp_*",
    ]

    print(f"Starting clean upload of '{project_root}' to Hugging Face {args.type}: '{args.repo}'...")
    print("Ignoring .venv, .git, logs, and large local datasets...")

    url = upload_folder(
        folder_path=project_root,
        repo_id=args.repo,
        repo_type=args.type,
        ignore_patterns=ignore_patterns,
    )

    print("\n============================================================")
    print("SUCCESS! Upload complete.")
    print(f"View live repository at: {url}")
    print("============================================================")

if __name__ == "__main__":
    main()
