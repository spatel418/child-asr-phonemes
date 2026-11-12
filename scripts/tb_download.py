"""Download TalkBank / PhonBank corpora straight to Quest.

Run it yourself in a terminal. Credentials never live in this repo:
  - email:    --email, or the TALKBANK_EMAIL environment variable
  - password: typed at the prompt (getpass), or TALKBANK_PASSWORD if already set
              in your shell. Never write it into this file.
Files go to $CHILD_ASR_RAW/<corpus>/ (or --out). Set it in your shell, e.g.
    export CHILD_ASR_RAW=/path/to/data/child_asr/raw

    PY=/projects/p32576/simron/envs/pasketti-phonetic/bin/python
    $PY -I scripts/tb_download.py --list PERCEPT-R          # dry run: log in, show what would download
    $PY -I scripts/tb_download.py PERCEPT-R ACAD            # download those corpora
    $PY -I scripts/tb_download.py --list all                # sizes for everything (except Providence)
    $PY -I scripts/tb_download.py all                       # everything below, in order (except Providence)

Re-running skips files that are already complete, so it is safe to restart.
"""
import argparse, getpass, os, re, shutil, sys, time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

AUTH = "https://sla2.talkbank.org"
OUT_ROOT = os.environ.get("CHILD_ASR_RAW", "data/child_asr/raw")  # set in your shell; data is never committed


# name: (transcript-zip path, media folder path, extra zips)
CORPORA = {
    "PERCEPT-R":       ("phon/Clinical/PERCEPT-R", "phon/Clinical/PERCEPT-R", ["phon/0extra/PERCEPT-R.zip"]),
    "ACAD":            ("phon/Eng-NA/ACAD", "phon/Eng-NA/ACAD", []),
    "PERCEPT-GFTA":    ("phon/Clinical/PERCEPT-GFTA", "phon/Clinical/PERCEPT-GFTA", ["phon/0extra/PERCEPT-GFTA.zip"]),
    "Preston":         ("phon/Clinical/Preston", "phon/Clinical/Preston", []),
    "TorringtonEaton": ("phon/Clinical/TorringtonEaton", "phon/Clinical/TorringtonEaton", []),
    "EllisWeismer":    ("childes/Clinical-Eng/EllisWeismer", "childes/Clinical-Eng/EllisWeismer", []),
    "Cameron":         ("childes/Eng-AAE/Cameron", "childes/Eng-AAE/Cameron", []),
    "OCSC":            ("childes/Eng-NA/OCSC", "childes/Eng-NA/OCSC", []),
    # added 2026-11-11: more PhonBank English corpora with audio + Phon (IPA) data
    "Cummings":        ("phon/Clinical/Cummings", "phon/Clinical/Cummings", []),
    # PCSC: transcripts only, no media on TalkBank
    "PCSC":            ("phon/Clinical/PCSC", "phon/Clinical/PCSC", []),
    "Penney":          ("phon/Eng-NA/Penney", "phon/Eng-NA/Penney", []),
    "Chiat":           ("phon/Clinical/Chiat", "phon/Clinical/Chiat", []),
    "McAllister":      ("phon/Clinical/McAllister", "phon/Clinical/McAllister", []),
    "PaidoEnglish":    ("phon/Eng-NA/PaidoEnglish", "phon/Eng-NA/PaidoEnglish", []),
    "Providence":      ("phon/Eng-NA/Providence", "phon/Eng-NA/Providence", []),  # video, very large
}
# 'all' skips these (too big for our disk); name them explicitly to get them
NOT_IN_ALL = {"Providence"}
MIN_FREE = 20e9  # stop if the project disk would drop below 20 GB free
MEDIA_EXT = (".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aif", ".aiff")


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.hrefs.append(v)


def login(s, email):
    pswd = os.environ.get("TALKBANK_PASSWORD") or getpass.getpass(f"TalkBank password for {email}: ")
    r = s.post(f"{AUTH}/logInUser", json={"email": email, "pswd": pswd}, timeout=60)
    del pswd
    try:
        res = r.json()
    except ValueError:
        sys.exit(f"Login failed: unexpected reply (HTTP {r.status_code}).")
    if not res.get("success"):
        sys.exit(f"Login failed: {res.get('respMsg')}")
    log("Logged in.")


LOG_FH = None
DEBUG_DIR = None


def log(msg):
    print(msg, flush=True)
    if LOG_FH:
        LOG_FH.write(msg + "\n"); LOG_FH.flush()


def is_html(resp):
    return "text/html" in resp.headers.get("Content-Type", "")


def total_size(r):
    """Full file size. The media server answers 206 with 'Content-Range: bytes 0-N/TOTAL'."""
    cr = r.headers.get("Content-Range", "")
    if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
        return int(cr.rsplit("/", 1)[1])
    return int(r.headers.get("Content-Length") or 0)


def get_file(s, url, dest, list_only):
    """Download url to dest unless already complete. Returns size in bytes or None.
    Accepts 200 and 206 (this server sends 206 even without a Range request) and
    keeps asking for the rest until the file is complete."""
    r = s.get(url, stream=True, timeout=120, allow_redirects=True)
    size = total_size(r)
    if r.status_code not in (200, 206) or (is_html(r) and not dest.endswith((".html", ".htm"))):
        log(f"  NOT AVAILABLE (HTTP {r.status_code}, {r.headers.get('Content-Type')}): {url}")
        r.close(); return None
    if list_only:
        head = next(r.iter_content(4), b"")
        r.close()
        if dest.endswith(".zip"):
            if not head.startswith(b"PK"):
                log(f"  WARNING: not a zip (starts {head!r}): {url}")
            elif not size:
                log(f"  ok (real zip, size not reported until downloaded): {url}")
        return size
    if os.path.exists(dest) and size and os.path.getsize(dest) == size:
        r.close(); return size
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if shutil.disk_usage(OUT_ROOT).free - size < MIN_FREE:
        r.close(); sys.exit(f"STOPPED: less than {MIN_FREE/1e9:.0f} GB would be left on disk. Re-run later to resume.")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    r.close()
    tries = 0
    while size and os.path.getsize(tmp) < size and tries < 50:   # server sent only part: fetch the rest
        tries += 1
        got = os.path.getsize(tmp)
        r = s.get(url, stream=True, timeout=120, headers={"Range": f"bytes={got}-"})
        if r.status_code not in (200, 206):
            r.close(); break
        if r.status_code == 200:          # server ignored Range: it's the whole file again
            mode = "wb"
        else:
            mode = "ab"
        with open(tmp, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        r.close()
    got = os.path.getsize(tmp)
    if size and got != size:
        log(f"  INCOMPLETE ({got} of {size} bytes), kept as .part; re-run to retry: {url}")
        return None
    os.replace(tmp, dest)
    return got


def crawl_media(s, url, seen=None):
    """Yield media file URLs under a media.talkbank.org folder listing."""
    seen = seen if seen is not None else set()
    if url in seen:
        return
    seen.add(url)
    r = s.get(url, timeout=120)
    if r.status_code != 200 or not is_html(r):
        log(f"  cannot list {url} (HTTP {r.status_code})"); return
    if "initAuthModals" in r.text and "<a " not in r.text:
        log(f"  media listing needs authorization: {url}"); return
    if DEBUG_DIR:
        dump = os.path.join(DEBUG_DIR, "listing_" + re.sub(r"[^A-Za-z0-9]+", "_", urlparse(url).path) + ".html")
        with open(dump, "w") as f:
            f.write(r.text)
        log(f"  [debug] saved listing ({len(r.text)} chars) -> {dump}")
    p = Links(); p.feed(r.text)
    base = url if url.endswith("/") else url + "/"
    base_path = urlparse(base).path
    for h in p.hrefs:
        u = urlparse(urljoin(base, h))
        if u.hostname != "media.talkbank.org" or u.query:
            continue  # other sites, sort links
        path = re.sub(r"/+", "/", "/" + u.path)  # site writes folder links like .../Cummings//CHAT
        if not path.startswith(base_path) or path.rstrip("/") == base_path.rstrip("/"):
            continue  # parent links
        last = path.rstrip("/").rsplit("/", 1)[-1]
        if path.lower().endswith(MEDIA_EXT):
            yield f"https://media.talkbank.org{path}"
        elif path.endswith("/") or "." not in last:
            yield from crawl_media(s, f"https://media.talkbank.org{path.rstrip('/')}/", seen)  # folder


def do_corpus(s, name, list_only):
    data_path, media_path, extras = CORPORA[name]
    out = os.path.join(OUT_ROOT, name)
    log(f"\n=== {name} -> {out}")
    total = 0
    sz = get_file(s, f"https://talkbank.org/data/{data_path}?f=zip",
                  os.path.join(out, "transcripts", f"{name}.zip"), list_only)
    log(f"  transcripts zip: {sz and round(sz/1e6, 1)} MB"); total += sz or 0
    for ex in extras:
        sz = get_file(s, f"https://media.talkbank.org/{ex}",
                      os.path.join(out, "extra", os.path.basename(ex)), list_only)
        log(f"  extra {os.path.basename(ex)}: {sz and round(sz/1e9, 2)} GB"); total += sz or 0
    media_root = f"https://media.talkbank.org/{media_path}/"
    files = list(crawl_media(s, media_root))
    log(f"  media files found: {len(files)}")
    t0 = time.time()
    for i, u in enumerate(files, 1):
        rel = urlparse(u).path.split(f"/{media_path}/", 1)[-1]
        sz = get_file(s, u, os.path.join(out, "media", rel), list_only)
        total += sz or 0
        if i % 50 == 0 or i == len(files):
            log(f"  {i}/{len(files)} files, {total/1e9:.2f} GB, {time.time()-t0:.0f}s")
    log(f"  {name} total: {total/1e9:.2f} GB {'(would download)' if list_only else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+", help=f"'all' or any of: {', '.join(CORPORA)}")
    ap.add_argument("--list", action="store_true", help="log in and report sizes; download nothing")
    ap.add_argument("--email", default=os.environ.get("TALKBANK_EMAIL"), help="or set TALKBANK_EMAIL")
    ap.add_argument("--out", default=None, help="output root (default $CHILD_ASR_RAW)")
    ap.add_argument("--debug", action="store_true", help="save raw media listing pages next to the log")
    a = ap.parse_args()
    global LOG_FH, DEBUG_DIR, OUT_ROOT
    if not a.email:
        sys.exit("Give --email or set TALKBANK_EMAIL.")
    if a.out:
        OUT_ROOT = a.out
    logdir = os.path.join(OUT_ROOT, "_download_logs")
    os.makedirs(logdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    LOG_FH = open(os.path.join(logdir, f"tb_download_{stamp}.log"), "w")
    if a.debug:
        DEBUG_DIR = os.path.join(logdir, f"debug_{stamp}"); os.makedirs(DEBUG_DIR, exist_ok=True)
    log(f"log file: {LOG_FH.name}  args: {' '.join(sys.argv[1:])}")
    names = [n for n in CORPORA if n not in NOT_IN_ALL] if a.corpora == ["all"] else a.corpora
    bad = [n for n in names if n not in CORPORA]
    if bad:
        sys.exit(f"Unknown corpus: {bad}")
    s = requests.Session()
    s.headers["User-Agent"] = "talkbank-download-script (research)"
    login(s, a.email)
    for n in names:
        do_corpus(s, n, a.list)


if __name__ == "__main__":
    main()
