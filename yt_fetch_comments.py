import os, re, json, argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import pandas as pd
from googleapiclient.discovery import build

API_KEY = os.getenv("YT_API_KEY")

def extract_video_id(inp: str) -> str:
    s = (inp or "").strip().strip('"').strip("'")
    # raw 11-char id?
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    try:
        u = urlparse(s)
        if "youtu.be" in u.netloc:
            cand = u.path.strip("/").split("/")[0]
            return cand if re.fullmatch(r"[A-Za-z0-9_-]{11}", cand) else ""
        qs = parse_qs(u.query)
        cand = (qs.get("v") or [""])[0]
        return cand if re.fullmatch(r"[A-Za-z0-9_-]{11}", cand) else ""
    except Exception:
        return ""

def fetch_all_comments(
    video_id: str,
    order: str = "time",
    max_top_level: int | None = None,
    no_replies: bool = False,
    max_total: int | None = None,
    save_state_path: str | None = None,
    resume: bool = False,
    checkpoint_interval: int = 500,
):
    """Resumable, quota-safe fetcher of top-level comments (+ replies unless no_replies)."""
    if not API_KEY:
        raise SystemExit("Please set YT_API_KEY environment variable.")
    yt = build("youtube", "v3", developerKey=API_KEY)

    # load/initialize state
    state = {
        "video_id": video_id,
        "order": order,
        "page_token": None,
        "processed_top_level": [],
        "current_top_id": None,
        "reply_page_token": None,
    }
    if resume and save_state_path and Path(save_state_path).exists():
        try:
            prior = json.loads(Path(save_state_path).read_text(encoding="utf-8"))
            if prior.get("video_id") == video_id and prior.get("order") == order:
                state.update(prior)
                print(f"Resuming from {save_state_path}")
            else:
                print("State file doesn't match video/order. Starting fresh.")
        except Exception as e:
            print(f"Could not read state ({e}). Starting fresh.")

    def save_state():
        if save_state_path:
            Path(save_state_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    total_rows = 0
    fetched_threads = 0

    def maybe_checkpoint():
        nonlocal total_rows
        if save_state_path and checkpoint_interval and (total_rows % checkpoint_interval == 0):
            save_state()

    # if we paused mid-replies, finish those first
    if state.get("current_top_id") and not no_replies:
        top_id = state["current_top_id"]
        reply_page = state.get("reply_page_token")
        try:
            while True:
                reply_resp = yt.comments().list(
                    part="snippet",
                    parentId=top_id,
                    maxResults=100,
                    pageToken=reply_page,
                    textFormat="plainText",
                ).execute()
                for r in reply_resp.get("items", []) or []:
                    rs = r["snippet"]
                    rows.append({
                        "video_id": video_id, "comment_id": r["id"], "parent_id": top_id,
                        "author": rs.get("authorDisplayName",""), "like_count": rs.get("likeCount",0),
                        "published_at": rs.get("publishedAt",""),
                        "updated_at": rs.get("updatedAt", rs.get("publishedAt","")),
                        "text": rs.get("textOriginal","")
                    })
                    total_rows += 1
                    maybe_checkpoint()
                    if max_total and total_rows >= max_total:
                        save_state(); return rows
                reply_page = reply_resp.get("nextPageToken")
                state["reply_page_token"] = reply_page
                if not reply_page: break
        except Exception as e:
            if "quotaExceeded" in str(e):
                print("Quota hit while resuming replies. Saved state; returning partial results.")
                save_state(); return rows
            raise
        state["processed_top_level"] = list(set(state.get("processed_top_level", [])) | {top_id})
        state["current_top_id"] = None
        state["reply_page_token"] = None
        save_state()

    page_token = state.get("page_token")
    seen_tokens = set()

    while True:
        try:
            resp = yt.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                order=order,
                textFormat="plainText",
                pageToken=page_token
            ).execute()
        except Exception as e:
            msg = str(e)
            if "quotaExceeded" in msg:
                print("Daily quota hit. Saved state; returning partial results.")
                state["page_token"] = page_token; save_state(); return rows
            if page_token and ("processingFailure" in msg or "invalidPageToken" in msg):
                page_token = None; continue
            raise

        items = resp.get("items", [])
        for it in items:
            top = it["snippet"]["topLevelComment"]; ts = top["snippet"]; top_id = top["id"]

            if top_id in set(state.get("processed_top_level", [])):
                continue

            rows.append({
                "video_id": video_id, "comment_id": top_id, "parent_id": None,
                "author": ts.get("authorDisplayName",""), "like_count": ts.get("likeCount",0),
                "published_at": ts.get("publishedAt",""),
                "updated_at": ts.get("updatedAt", ts.get("publishedAt","")),
                "text": ts.get("textOriginal","")
            })
            total_rows += 1; fetched_threads += 1; maybe_checkpoint()
            if max_top_level and fetched_threads >= max_top_level:
                state["page_token"] = resp.get("nextPageToken"); save_state(); return rows
            if max_total and total_rows >= max_total:
                state["page_token"] = resp.get("nextPageToken"); save_state(); return rows

            if not no_replies:
                # replies included in thread page
                for r in it.get("replies", {}).get("comments", []) or []:
                    rs = r["snippet"]
                    rows.append({
                        "video_id": video_id, "comment_id": r["id"], "parent_id": top_id,
                        "author": rs.get("authorDisplayName",""), "like_count": rs.get("likeCount",0),
                        "published_at": rs.get("publishedAt",""),
                        "updated_at": rs.get("updatedAt", rs.get("publishedAt","")),
                        "text": rs.get("textOriginal","")
                    })
                    total_rows += 1; maybe_checkpoint()
                    if max_total and total_rows >= max_total:
                        state["page_token"] = resp.get("nextPageToken"); save_state(); return rows

                # full replies pagination
                if it["snippet"].get("totalReplyCount", 0):
                    state["current_top_id"] = top_id; state["reply_page_token"] = None; save_state()
                    reply_page = None
                    try:
                        while True:
                            reply_resp = yt.comments().list(
                                part="snippet",
                                parentId=top_id,
                                maxResults=100,
                                pageToken=reply_page,
                                textFormat="plainText",
                            ).execute()
                            for r in reply_resp.get("items", []) or []:
                                rs = r["snippet"]
                                rows.append({
                                    "video_id": video_id, "comment_id": r["id"], "parent_id": top_id,
                                    "author": rs.get("authorDisplayName",""), "like_count": rs.get("likeCount",0),
                                    "published_at": rs.get("publishedAt",""),
                                    "updated_at": rs.get("updatedAt", rs.get("publishedAt","")),
                                    "text": rs.get("textOriginal","")
                                })
                                total_rows += 1; maybe_checkpoint()
                                if max_total and total_rows >= max_total:
                                    state["reply_page_token"] = reply_resp.get("nextPageToken"); save_state(); return rows
                            reply_page = reply_resp.get("nextPageToken")
                            state["reply_page_token"] = reply_page
                            if not reply_page: break
                    except Exception as e:
                        if "quotaExceeded" in str(e):
                            print("Quota hit during replies. Saved state; returning partial results.")
                            save_state(); return rows
                        raise
                    state["processed_top_level"] = list(set(state.get("processed_top_level", [])) | {top_id})
                    state["current_top_id"] = None; state["reply_page_token"] = None; save_state()

        next_tok = resp.get("nextPageToken")
        if not next_tok: break
        if next_tok in seen_tokens: page_token = None; continue
        seen_tokens.add(next_tok); page_token = next_tok

    return rows

def main():
    ap = argparse.ArgumentParser(description="Fetch all YouTube comments (with optional resume).")
    ap.add_argument("video", help="YouTube URL or 11-char ID")
    ap.add_argument("--order", choices=["time","relevance"], default="time")
    ap.add_argument("--csv", default="comments.csv", help="Output CSV")
    ap.add_argument("--no-replies", action="store_true", help="Fetch only top-level comments")
    ap.add_argument("--max-top-level", type=int, default=None)
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--save-state", default=None, help="JSON path to save/resume state")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    args = ap.parse_args()

    vid = extract_video_id(args.video)
    if not vid:
        raise SystemExit(f"Could not parse a valid 11-char video ID from input: {args.video}")
    print(f"Using videoId: {vid}")

    rows = fetch_all_comments(
        vid, order=args.order, max_top_level=args.max_top_level,
        no_replies=args.no_replies, max_total=args.max_total,
        save_state_path=args.save_state, resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )
    if not rows:
        print("No comments found or comments disabled.")
        return

    # Deduplicate by comment_id (in case of resumes)
    seen = set(); clean = []
    for r in rows:
        if r["comment_id"] in seen: continue
        seen.add(r["comment_id"]); clean.append(r)

    df = pd.DataFrame(clean, columns=[
        "video_id","comment_id","parent_id","author","like_count","published_at","updated_at","text"
    ])
    df.to_csv(args.csv, index=False, encoding="utf-8")
    n_top = sum(1 for r in clean if r["parent_id"] is None)
    n_rep = len(clean) - n_top
    print(f"Saved {len(clean)} rows → {args.csv} (top-level: {n_top}, replies: {n_rep})")

if __name__ == "__main__":
    main()
