"""Parallel chunked downloader for MNIST IDX files."""
import concurrent.futures as cf
import os
import sys
import urllib.request

FILES = {
    "train-images-idx3-ubyte.gz": None,
    "train-labels-idx1-ubyte.gz": None,
    "t10k-images-idx3-ubyte.gz": None,
    "t10k-labels-idx1-ubyte.gz": None,
}

SOURCES = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/{}",
    "https://raw.githubusercontent.com/fgnt/mnist/master/{}",
    "https://ossci-datasets.s3.amazonaws.com/mnist/{}",
]

CHUNK = 1 << 20  # 1 MiB
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return start, r.read()


def download(fname):
    out = os.path.join(OUT_DIR, fname)
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f"[skip] {fname} exists", flush=True)
        return
    last_err = None
    for tpl in SOURCES:
        url = tpl.format(fname)
        try:
            total = head_size(url)
            ranges = [(s, min(s + CHUNK - 1, total - 1)) for s in range(0, total, CHUNK)]
            buf = bytearray(total)
            with cf.ThreadPoolExecutor(max_workers=12) as ex:
                futs = [ex.submit(fetch_range, url, s, e) for s, e in ranges]
                for fut in cf.as_completed(futs):
                    s, data = fut.result()
                    buf[s:s + len(data)] = data
            with open(out, "wb") as f:
                f.write(bytes(buf))
            print(f"[ok] {fname} {total} bytes <- {url}", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[fail] {fname} <- {url}: {e}", flush=True)
    raise RuntimeError(f"all sources failed for {fname}: {last_err}")


def main():
    for fname in FILES:
        download(fname)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
