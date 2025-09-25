import sys, argparse, subprocess, os, shutil

def main():
    ap = argparse.ArgumentParser(description="Run fetch → cluster pipeline for one video.")
    ap.add_argument("video", help="YouTube URL or 11-char ID")
    ap.add_argument("--base", default="out", help="Base name for outputs (default: out)")
    ap.add_argument("--order", choices=["time","relevance"], default="time")
    ap.add_argument("--max-top-level", type=int, default=None)
    ap.add_argument("--no-replies", action="store_true")
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--save-state", default="state.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    ap.add_argument("--sim", type=float, default=0.88)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    raw_csv = f"{args.base}_raw.csv"
    fetch_cmd = [
        sys.executable, "fetch_comments.py", args.video,
        "--order", args.order, "--csv", raw_csv, "--checkpoint-interval", str(args.checkpoint_interval),
        "--save-state", args.save_state
    ]
    if args.no_replies: fetch_cmd.append("--no-replies")
    if args.resume: fetch_cmd.append("--resume")
    if args.max_top_level is not None:
        fetch_cmd += ["--max-top-level", str(args.max_top_level)]
    if args.max_total is not None:
        fetch_cmd += ["--max-total", str(args.max_total)]

    print(">>> Running fetch step …")
    r = subprocess.run(fetch_cmd)
    if r.returncode != 0:
        print("Fetch step failed."); sys.exit(r.returncode)

    if not os.path.exists(raw_csv):
        print("No raw CSV produced; exiting."); sys.exit(1)

    cluster_cmd = [
        sys.executable, "cluster_comments.py", raw_csv,
        "--csv-base", args.base, "--sim", str(args.sim), "--min-samples", str(args.min_samples),
        "--model", args.model
    ]
    print(">>> Running cluster step …")
    r2 = subprocess.run(cluster_cmd)
    sys.exit(r2.returncode)

if __name__ == "__main__":
    main()
