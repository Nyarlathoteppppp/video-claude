"""Look up Wikimedia Commons / Wikipedia files: real download URL, size, licence and author (stdlib only).

Guessed upload.wikimedia.org paths are often wrong; the API gives the real one, plus the licence to write into
ai_history/images/LICENSES.md.

    python tools/wiki_image.py "File:Alan Turing Aged 16.jpg" "File:Eniac.jpg"
    python tools/wiki_image.py --site en.wikipedia.org "File:DALL-E radish.jpg" --download ai_history/images/press
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request

UA = "video-claude/1.0 (https://github.com/Nyarlathoteppppp/video-claude)"   # Wikimedia asks for a descriptive UA


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return r.read()


def lookup(site, titles):
    q = urllib.parse.urlencode({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
                                "iiprop": "url|size|extmetadata", "format": "json"})
    pages = json.loads(get(f"https://{site}/w/api.php?{q}"))["query"]["pages"].values()
    for p in pages:
        info = (p.get("imageinfo") or [None])[0]
        if not info:
            yield {"title": p.get("title"), "missing": True}
            continue
        m = info.get("extmetadata", {})
        hidden = r"<(\w+)[^>]*display:\s*none[^>]*>.*?</\1>"            # e.g. machine-readable dates
        text = lambda k: re.sub(r"<[^>]+>", "", re.sub(hidden, "", m.get(k, {}).get("value", ""), flags=re.S)).strip()
        yield {"title": p["title"], "url": info["url"].split("?")[0], "size": f"{info['width']}x{info['height']}",
               "license": text("LicenseShortName"), "artist": text("Artist"), "date": text("DateTimeOriginal"),
               "page": info.get("descriptionurl", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="+", help='file titles, e.g. "File:Eniac.jpg"')
    ap.add_argument("--site", default="commons.wikimedia.org")
    ap.add_argument("--download", metavar="DIR", help="also save each file into DIR")
    a = ap.parse_args()
    for r in lookup(a.site, a.titles):
        if r.get("missing"):
            print(f"{r['title']}: not found on {a.site} (try --site en.wikipedia.org)")
            continue
        print(f"{r['title']}\n  {r['url']}\n  {r['size']} | {r['license']} | {r['artist']} | {r['date']}\n  {r['page']}")
        if a.download:
            os.makedirs(a.download, exist_ok=True)
            out = os.path.join(a.download, os.path.basename(urllib.parse.unquote(r["url"])))
            with open(out, "wb") as fh:
                fh.write(get(r["url"]))
            print("  saved", out)


if __name__ == "__main__":
    main()
