"""Modul crawling komentar YouTube menggunakan YouTube Data API v3."""
import os
import re
import pandas as pd
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def build_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


# ==================== BARU: EKSTRAK VIDEO ID ====================
def extract_video_id(url: str) -> str:
    """Ekstrak video ID dari berbagai format URL YouTube."""
    if not url:
        return ""
    url = url.strip()

    # Format youtu.be/VIDEOID
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]

    # Format youtube.com/watch?v=VIDEOID
    parsed = urlparse(url)
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # Format youtube.com/embed/VIDEOID atau /shorts/VIDEOID
        m = re.search(r"/(embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(2)

    # Jika input hanya video ID (11 karakter)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    return ""


def get_video_info(youtube, video_id: str) -> dict:
    """Ambil metadata video."""
    try:
        resp = youtube.videos().list(
            part="snippet,statistics", id=video_id
        ).execute()
        if not resp.get("items"):
            return {}
        item = resp["items"][0]
        s, st = item["snippet"], item.get("statistics", {})
        return {
            "video_id": video_id,
            "title": s["title"],
            "channel": s["channelTitle"],
            "published_at": s["publishedAt"],
            "view_count": st.get("viewCount", 0),
            "like_count": st.get("likeCount", 0),
            "comment_count": st.get("commentCount", 0),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
    except HttpError as e:
        print(f"[ERROR] get_video_info: {e}")
        return {}


# ==================== MODIFIKASI: AMBIL REPLY ====================
def get_replies(youtube, parent_comment_id: str, max_pages: int = 2) -> list:
    """Mengambil reply dari sebuah komentar top-level."""
    replies = []
    next_page_token = None
    page = 0
    try:
        while page < max_pages:
            resp = youtube.comments().list(
                part="snippet",
                parentId=parent_comment_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()

            for item in resp.get("items", []):
                s = item["snippet"]
                replies.append({
                    "comment_id": item["id"],
                    "parent_id": parent_comment_id,
                    "type": "reply",
                    "author": s.get("authorDisplayName", ""),
                    "comment_text": s.get("textDisplay", ""),
                    "published_at": s.get("publishedAt", ""),
                    "like_count": s.get("likeCount", 0),
                })
            next_page_token = resp.get("nextPageToken")
            page += 1
            if not next_page_token:
                break
    except HttpError as e:
        print(f"[WARN] get_replies {parent_comment_id}: {e}")
    return replies


def get_comments_with_replies(youtube, video_id: str, max_pages: int = 3,
                              include_replies: bool = True,
                              max_reply_pages: int = 2) -> list:
    """Ambil komentar top-level + seluruh reply."""
    all_comments = []
    next_page_token = None
    page = 0
    try:
        while page < max_pages:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            )
            response = request.execute()

            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                parent_id = item["id"]

                # Komentar top-level
                all_comments.append({
                    "comment_id": parent_id,
                    "parent_id": None,
                    "type": "comment",
                    "video_id": video_id,
                    "author": top.get("authorDisplayName", ""),
                    "comment_text": top.get("textDisplay", ""),
                    "published_at": top.get("publishedAt", ""),
                    "like_count": top.get("likeCount", 0),
                    "reply_count": item["snippet"].get("totalReplyCount", 0),
                })

                # Reply (jika ada)
                if include_replies and item["snippet"].get("totalReplyCount", 0) > 0:
                    replies = get_replies(
                        youtube, parent_id, max_pages=max_reply_pages
                    )
                    for r in replies:
                        r["video_id"] = video_id
                        r["reply_count"] = 0
                        all_comments.append(r)

            next_page_token = response.get("nextPageToken")
            page += 1
            if not next_page_token:
                break
    except HttpError as e:
        print(f"[WARN] get_comments_with_replies {video_id}: {e}")
    return all_comments


# ==================== BARU: CRAWL DARI SATU URL ====================
def crawl_from_url(api_key: str, url: str, max_pages: int = 5,
                   include_replies: bool = True,
                   comments_csv: str = None, videos_csv: str = None):
    """Crawl komentar + reply dari SATU URL YouTube."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("URL YouTube tidak valid. Contoh: https://www.youtube.com/watch?v=XXXX")

    youtube = build_youtube_client(api_key)

    # Metadata video
    video_info = get_video_info(youtube, video_id)
    if not video_info:
        raise ValueError("Video tidak ditemukan atau tidak dapat diakses.")

    # Komentar + reply
    comments = get_comments_with_replies(
        youtube, video_id,
        max_pages=max_pages,
        include_replies=include_replies,
    )

    df_videos = pd.DataFrame([video_info])
    df_comments = pd.DataFrame(comments)

    if not df_comments.empty:
        df_comments = df_comments.drop_duplicates(subset=["comment_id"])

    if comments_csv:
        df_comments.to_csv(comments_csv, index=False, encoding="utf-8-sig")
    if videos_csv:
        df_videos.to_csv(videos_csv, index=False, encoding="utf-8-sig")

    return df_videos, df_comments


# ==================== FUNGSI LAMA (tetap dipertahankan) ====================
def search_videos(youtube, query: str, max_results: int = 5):
    """Mencari video berdasarkan kata kunci."""
    videos = []
    try:
        request = youtube.search().list(
            q=query, part="id,snippet", maxResults=max_results,
            type="video", relevanceLanguage="id",
        )
        response = request.execute()
        for item in response.get("items", []):
            if item["id"].get("kind") != "youtube#video":
                continue
            vid = item["id"]["videoId"]
            snippet = item["snippet"]
            stats_resp = youtube.videos().list(
                part="statistics", id=vid
            ).execute()
            stats = stats_resp["items"][0]["statistics"] if stats_resp["items"] else {}
            videos.append({
                "video_id": vid, "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
                "view_count": stats.get("viewCount", 0),
                "like_count": stats.get("likeCount", 0),
                "comment_count": stats.get("commentCount", 0),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
    except HttpError as e:
        print(f"[ERROR] search_videos: {e}")
    return videos


def crawl_mbg_data(api_key: str, queries: list, max_videos: int,
                   max_pages: int, videos_csv: str, comments_csv: str):
    """Pipeline crawling lama berdasarkan kata kunci."""
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY belum di-set.")
    youtube = build_youtube_client(api_key)
    all_videos, all_comments = [], []
    for q in queries:
        vids = search_videos(youtube, q, max_results=max_videos)
        all_videos.extend(vids)
        for v in vids:
            cmts = get_comments_with_replies(
                youtube, v["video_id"], max_pages=max_pages, include_replies=True
            )
            all_comments.extend(cmts)

    df_videos = pd.DataFrame(all_videos).drop_duplicates(subset=["video_id"])
    df_comments = pd.DataFrame(all_comments).drop_duplicates(subset=["comment_id"])
    df_videos.to_csv(videos_csv, index=False, encoding="utf-8-sig")
    df_comments.to_csv(comments_csv, index=False, encoding="utf-8-sig")
    return df_videos, df_comments