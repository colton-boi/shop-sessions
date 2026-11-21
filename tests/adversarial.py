#!/usr/bin/env python3
"""adversarial.py SRC DST: rewrite a generated dataset so every fast path in the pipeline meets its
fallback, while staying inside rules.md. IDs are remapped consistently across events, clicks and
links: some over 9 bytes, some multi-byte UTF-8, some upper-cased. Some F3 timestamps get leading
zeros up to 20 and 25 digits. Clicks gain equal-ts ties across several campaigns. Some lines end in
\\r. Extra files: header-only, no trailing newline, and sizes on exact page multiples."""
import hashlib, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]


def h(s, salt):
    return int.from_bytes(hashlib.blake2b((salt + s).encode(), digest_size=4).digest(), "little") / 2**32


def remap_user(u):
    if not u:
        return u
    x = h(u, "u")
    if x < 0.03:
        return u + "_extralong"
    if x < 0.04:
        return "ü" + u
    if x < 0.045:
        return u.upper()
    return u


def remap_id(e, salt):
    if e and h(e, salt) < 0.02:
        return e + "xyzw0123"
    return e


def remap_ts(t):
    if t.isdigit() and h(t, "t") < 0.05:
        return t.zfill(20 if h(t, "w") < 0.5 else 25)
    return t


def split(line):
    """Fields and whether each was quoted, or None for a line to leave untouched."""
    out = []
    for f in line.split(","):
        if len(f) >= 2 and f[0] == '"' and f[-1] == '"' and '"' not in f[1:-1]:
            out.append((f[1:-1], True))
        elif '"' in f:
            return None
        else:
            out.append((f, False))
    return out


def join(fields):
    return ",".join(f'"{v}"' if q else v for v, q in fields)


campaigns = [l.split(",")[0] for l in open(os.path.join(src, "campaigns.csv")).read().splitlines()[1:]]
shutil.rmtree(dst, ignore_errors=True)
os.makedirs(dst)
for f in ("products.csv", "campaigns.csv", "manifest.json"):
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))

for kind, n, edit in (
    ("events", 7, lambda fs: [remap_id(fs[0], "e"), remap_user(fs[1]), remap_ts(fs[2])] + fs[3:]),
    ("clicks", 4, lambda fs: [remap_id(fs[0], "k"), remap_user(fs[1]), fs[2], remap_ts(fs[3])]),
    ("links", 2, lambda fs: [remap_user(fs[0]), remap_user(fs[1])]),
):
    os.makedirs(os.path.join(dst, kind))
    names = sorted(os.listdir(os.path.join(src, kind)))
    for k, name in enumerate(names):
        text = open(os.path.join(src, kind, name), encoding="utf-8", newline="").read()
        lines = text.split("\n")
        trailing = lines[-1] == ""
        if trailing:
            lines.pop()
        out = [lines[0]] if lines else []
        for line in lines[1:]:
            fs = split(line)
            if fs is None or len(fs) != n:
                out.append(line)
                continue
            vals = edit([v for v, _ in fs])
            new = [(v, q) for v, (_, q) in zip(vals, fs)]
            row = join(new)
            # Decided per (event_id, ingest_ts) and per click_id, so copies stay identical as the rules promise.
            if kind == "events" and h(vals[0] + "|" + vals[3], "r") < 0.005:
                row += "\r"
            out.append(row)
            if kind == "clicks" and h(vals[0], "tie") < 0.02 and fs[0][0] and fs[1][0]:
                # Same person, same ts, several campaigns, click_ids on both sides of the original.
                for pre in ("a", "z", "m"):
                    c = campaigns[int(h(pre + vals[0], "c") * len(campaigns))]
                    out.append(join([(pre + vals[0], False), (vals[1], fs[1][1]), (c, False), (vals[3], False)]))
        body = "\n".join(out)
        if k % 5 == 1:
            # Pad the last line's last field so the file ends exactly on a page boundary, no newline.
            body += "x" * ((-len(body.encode())) % 4096)
        elif k % 5 != 2:
            body += "\n"
        open(os.path.join(dst, kind, name), "w", encoding="utf-8", newline="").write(body)
    header = open(os.path.join(src, kind, names[0])).readline()
    open(os.path.join(dst, kind, "zz-header-only.csv"), "w").write(header)
    open(os.path.join(dst, kind, "zz-header-no-newline.csv"), "w").write(header.rstrip("\n"))
