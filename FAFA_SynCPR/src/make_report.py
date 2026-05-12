"""
Tạo file report.html từ thư mục result/ có sẵn.
Với query sai, hiển thị thêm ảnh ground truth đúng.

Chạy:
    python make_report.py \
        --result-dir ../../result \
        --itcpr-root ../../ITCPR
"""
import argparse, base64, re, json
from pathlib import Path
from io import BytesIO
from PIL import Image


def img_b64(path, max_h=220) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.resize((int(w * max_h / h), max_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def parse_info(info_path: Path) -> dict:
    out = {}
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if "Hit@1"        in line: out["hit1"]    = "YES" in line
        elif "Hit@5"      in line: out["hit5"]     = "YES" in line
        elif "Hit@10"     in line: out["hit10"]    = "YES" in line
        elif "Instance ID" in line:
            out["iid"] = int(line.split(":", 1)[-1].strip())
        elif line.startswith("Correct answer"):
            out["correct_ranks"] = line.split(":", 1)[-1].strip()
    return out


def load_gallery_index(itcpr_root: Path) -> dict:
    """iid → list of absolute image paths (bỏ distractor iid=-1)."""
    gallery_json = itcpr_root / "datasets" / "gallery.json"
    img_dir      = itcpr_root / "datasets"
    index = {}
    for item in json.loads(gallery_json.read_text()):
        iid = item["instance_id"]
        if iid == -1:
            continue
        index.setdefault(iid, []).append(str(img_dir / item["file_path"]))
    return index


def build_html(result_dir: Path, gallery_index: dict) -> str:
    summary_text  = (result_dir / "summary.txt").read_text(encoding="utf-8")
    query_folders = sorted(result_dir.glob("query_*/"), key=lambda p: p.name)

    cards = []
    for folder in query_folders:
        caption = (folder / "caption.txt").read_text(encoding="utf-8").strip()
        info    = parse_info(folder / "info.txt")
        ref_b64 = img_b64(folder / "reference.jpg")

        is_miss = not (info.get("hit1") or info.get("hit5") or info.get("hit10"))

        hit_label = (
            '<span class="hit">✓ Hit@1</span>'   if info.get("hit1")  else
            '<span class="hit5">✓ Hit@5</span>'  if info.get("hit5")  else
            '<span class="hit10">✓ Hit@10</span>') if not is_miss else (
            '<span class="miss">✗ Miss</span>'
        )

        # ── ground truth block (chỉ hiện khi sai) ────────────────────
        gt_html = ""
        if is_miss and gallery_index:
            iid = info.get("iid")
            gt_paths = gallery_index.get(iid, []) if iid else []
            if gt_paths:
                gt_cells = "".join(
                    f'<div class="gt-cell">'
                    f'<img src="data:image/jpeg;base64,{img_b64(p, max_h=160)}">'
                    f'</div>'
                    for p in gt_paths[:4]          # tối đa 4 ảnh đúng
                )
                gt_html = f"""
    <div class="gt-block">
      <div class="gt-title">Ground Truth</div>
      <div class="gt-row">{gt_cells}</div>
    </div>"""

        # ── top-10 retrieved ──────────────────────────────────────────
        rank_files = sorted(folder.glob("rank_*.jpg"),
                            key=lambda p: int(re.search(r"rank_(\d+)", p.name).group(1)))
        rank_cells = []
        for rf in rank_files:
            rank_no   = int(re.search(r"rank_(\d+)", rf.name).group(1))
            correct   = "CORRECT" in rf.name
            rank_cells.append(
                f'<div class="rank-cell {"correct-border" if correct else "wrong-border"}">'
                f'<img src="data:image/jpeg;base64,{img_b64(rf)}">'
                f'<div class="rank-label">Rank {rank_no}{" ✓" if correct else ""}</div>'
                f'</div>'
            )

        cards.append(f"""
<div class="card">
  <div class="card-header">
    <b>{folder.name.replace("_"," ").title()}</b>
    &nbsp;{hit_label}
    &nbsp;<span class="correct-ranks">Correct at: {info.get("correct_ranks","—")}</span>
  </div>
  <div class="card-body">
    <div class="ref-col">
      <img class="ref-img" src="data:image/jpeg;base64,{ref_b64}">
      <div class="caption-box">{caption}</div>
      <div class="ref-label">Reference</div>
    </div>
    <div class="arrow">➜</div>
    <div class="right-col">
      {gt_html}
      <div class="ranks-row">{"".join(rank_cells)}</div>
    </div>
  </div>
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ITCPR Retrieval Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI",Arial,sans-serif;background:#0f0f1a;color:#dde;padding:24px}}
  h1{{color:#e94560;margin-bottom:8px}}
  .summary{{background:#1a1a2e;border-left:4px solid #e94560;padding:12px 18px;
            border-radius:6px;font-size:.88em;line-height:1.7;
            display:inline-block;margin-bottom:28px}}
  .card{{background:#16213e;border-radius:12px;padding:16px;margin-bottom:22px;
         box-shadow:0 4px 14px rgba(0,0,0,.4)}}
  .card-header{{font-size:.95em;margin-bottom:12px;display:flex;
                align-items:center;gap:10px;flex-wrap:wrap}}
  .hit  {{background:#2ecc71;color:#000;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:bold}}
  .hit5 {{background:#f39c12;color:#000;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:bold}}
  .hit10{{background:#3498db;color:#fff;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:bold}}
  .miss {{background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:.82em;font-weight:bold}}
  .correct-ranks{{font-size:.82em;color:#aaa}}
  .card-body{{display:flex;align-items:flex-start;gap:14px}}
  .ref-col{{display:flex;flex-direction:column;align-items:center;min-width:120px;max-width:140px}}
  .ref-img{{max-height:220px;max-width:130px;border-radius:8px;border:3px solid #e94560}}
  .caption-box{{font-size:.76em;color:#bbb;margin-top:8px;text-align:center;line-height:1.4}}
  .ref-label{{font-size:.8em;color:#e94560;font-weight:bold;margin-top:4px}}
  .arrow{{font-size:2.4em;color:#e94560;align-self:center;flex-shrink:0}}
  .right-col{{display:flex;flex-direction:column;gap:10px}}
  .gt-block{{background:#0d2b1f;border:2px solid #2ecc71;border-radius:8px;padding:8px 10px}}
  .gt-title{{font-size:.78em;color:#2ecc71;font-weight:bold;margin-bottom:6px}}
  .gt-row{{display:flex;gap:6px;flex-wrap:wrap}}
  .gt-cell img{{max-height:140px;max-width:80px;border-radius:5px;
                border:2px solid #2ecc71;display:block}}
  .ranks-row{{display:flex;flex-wrap:wrap;gap:8px}}
  .rank-cell{{text-align:center;border-radius:8px;border:3px solid #555;padding:5px;width:95px}}
  .rank-cell img{{max-height:160px;max-width:85px;border-radius:5px;display:block;margin:0 auto}}
  .correct-border{{border-color:#2ecc71}}
  .wrong-border{{border-color:#444}}
  .rank-label{{font-size:.75em;color:#ccc;margin-top:4px}}
</style>
</head>
<body>
<h1>ITCPR Retrieval Report</h1>
<div class="summary">{summary_text.replace(chr(10),"<br>")}</div>
{"".join(cards)}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",  default="../../result")
    parser.add_argument("--itcpr-root",  default="../../ITCPR",
                        help="Cần để tìm ảnh ground truth cho query sai")
    args = parser.parse_args()

    result_dir    = Path(args.result_dir)
    gallery_index = load_gallery_index(Path(args.itcpr_root))
    print(f"Gallery index loaded: {len(gallery_index)} instance IDs")

    html = build_html(result_dir, gallery_index)
    out  = result_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out.resolve()}")
    print(f"Copy về máy:  scp root@<ip>:{out.resolve()} .")


if __name__ == "__main__":
    main()
