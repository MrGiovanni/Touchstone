#!/usr/bin/env python3
"""
build.py — regenerate the Touchstone leaderboard from the repository README.

The leaderboard lives in the README as ten Markdown tables wrapped in shields.io
badge markup, which is fine for a human reader and useless to anything else.
This script turns it into data:

    data/leaderboard.json   one record per (model, organ), machine-readable
    data/leaderboard.csv    the same thing flat, for spreadsheets and pandas
    data/models.csv         per-model metadata: org, params, speed, paper, code
    index.html              a static GitHub Pages leaderboard, no dependencies

Run from the leaderboard/ directory:

    python build.py ../README.md

Re-run it whenever the README tables change, and commit the result.  The README
stays the editable source of truth; this directory stays the machine-readable
mirror.
"""

import csv
import io
import json
import os
import re
import sys

# Paper-authoritative figures (NeurIPS 2024, arXiv 2411.03670). Kept here so the
# page and the README cannot drift apart silently again.
FACTS = {
    "train_volumes": 5195, "train_hospitals": 76,
    "test_volumes": 5903, "test_hospitals": 11,
    "organs": 9, "algorithms": 19, "teams": 14,
    # The paper benchmarks 19 algorithms from 14 teams. The README's leaderboard tables
    # publish DSC for 17 of them; STU-Net-H and STU-Net-L have result files under
    # totalsegmentator_results/ but no ranked row, so they are absent here. `ranked`
    # is what this page actually shows -- keep the two in step.
    "algorithms_ranked": 17,
}
ORGANS = ["Aorta", "Gallbladder", "KidneyL", "KidneyR", "Liver",
          "Pancreas", "Postcava", "Spleen", "Stomach"]


def strip_md(cell):
    """Reduce a badge-laden Markdown cell to its last URL, or to plain text."""
    urls = re.findall(r"\]\((https?://[^)]+)\)", cell)
    if urls:
        return urls[-1]
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip()


def parse_tables(md):
    """Yield (label, header, rows) for every Markdown table in the README."""
    tables, cur, ctx, pending = [], [], "Overall", None
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("<summary"):
            pending = re.sub(r"<[^>]+>", "", s).replace("\U0001F3C6", "").strip()
        if s.startswith("|"):
            if not cur and pending:
                ctx, pending = pending, None
            cur.append(s)
        else:
            if len(cur) >= 3:
                tables.append((ctx, cur))
            cur = []
    if len(cur) >= 3:
        tables.append((ctx, cur))

    out = []
    for label, rows in tables:
        header = [c.strip() for c in rows[0].strip("|").split("|")]
        recs = []
        for r in rows[2:]:
            cells = [c.strip() for c in r.strip("|").split("|")]
            if len(cells) < 3:
                continue
            recs.append({h.lower().replace(" ", "_").replace(".", ""): strip_md(c)
                         for h, c in zip(header, cells)})
        if recs:
            out.append((label, header, recs))
    return out


def build(readme_path):
    md = io.open(readme_path, encoding="utf-8").read()
    tables = parse_tables(md)

    overall, per_organ, meta = None, {}, None
    for label, header, recs in tables:
        cols = [h.lower() for h in header]
        if "parameter" in " ".join(cols):
            meta = recs
        elif label.startswith("Overall"):
            overall = recs
        else:
            organ = label.split(" - ")[0].strip()
            if organ in ORGANS:
                per_organ[organ] = recs

    if overall is None:
        raise SystemExit("could not find the overall leaderboard table")

    # ---- model registry ------------------------------------------------
    models = {}
    for r in overall:
        name = r["model"]
        models[name] = {
            "model": name,
            "organization": r.get("organization", ""),
            "average_dsc": r.get("average_dsc") or r.get("dsc", ""),
            "paper": r.get("paper", ""),
            "code": r.get("github", ""),
            "parameters": "", "inference_speed": "", "checkpoint": "",
        }
    for r in (meta or []):
        name = r["model"]
        if name in models:
            models[name].update({
                "parameters": r.get("parameter", ""),
                "inference_speed": r.get("infer_speed", ""),
                "checkpoint": r.get("download", ""),
            })

    # ---- long-form records ---------------------------------------------
    records = []
    for r in overall:
        records.append({"model": r["model"], "organ": "Average",
                        "metric": "DSC",
                        "value": float(r.get("average_dsc") or r.get("dsc") or 0),
                        "rank": r.get("rank", "")})
    for organ, recs in per_organ.items():
        for r in recs:
            try:
                v = float(r.get("dsc", ""))
            except ValueError:
                continue
            records.append({"model": r["model"], "organ": organ,
                            "metric": "DSC", "value": v, "rank": r.get("rank", "")})

    os.makedirs("data", exist_ok=True)
    payload = {"benchmark": "Touchstone 1.0", "facts": FACTS,
               "organs": ORGANS, "models": list(models.values()),
               "results": records}
    io.open("data/leaderboard.json", "w", encoding="utf-8").write(
        json.dumps(payload, indent=2))

    with io.open("data/leaderboard.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "organ", "metric", "value", "rank"])
        w.writeheader()
        w.writerows(records)

    with io.open("data/models.csv", "w", encoding="utf-8", newline="") as f:
        cols = ["model", "organization", "average_dsc", "parameters",
                "inference_speed", "paper", "code", "checkpoint"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: m.get(c, "") for c in cols} for m in models.values()])

    io.open("index.html", "w", encoding="utf-8").write(render(payload))

    print(f"models  : {len(models)}")
    print(f"organs  : {len(per_organ)}")
    print(f"records : {len(records)}")
    print("wrote   : data/leaderboard.json, data/leaderboard.csv, "
          "data/models.csv, index.html")


def render(payload):
    data = json.dumps(payload)
    f = payload["facts"]
    return """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Touchstone Benchmark Leaderboard</title>
<meta name="description" content="Touchstone 1.0: a large-scale benchmark for
medical image segmentation. """ + str(f["algorithms_ranked"]) + """ of """ + str(f["algorithms"]) + """ benchmarked AI algorithms, from """ + str(f["teams"]) + """ teams,
evaluated on """ + f"{f['test_volumes']:,}" + """ out-of-distribution CT volumes from """ + str(f["test_hospitals"]) + """ hospitals.">
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e5e5e5;--accent:#0b6bcb;
      --head:#f6f8fa;--hover:#f0f6ff;--gold:#b8860b}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--accent:#4493f8;
  --head:#161b22;--hover:#1c2733;--gold:#d4a72c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0 0 22px;max-width:64ch}
.facts{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 26px;padding:0;list-style:none}
.facts li{border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px}
.facts b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 16px}
.tabs button{border:1px solid var(--line);background:transparent;color:var(--fg);
  border-radius:999px;padding:6px 14px;font-size:13px;cursor:pointer}
.tabs button[aria-selected=true]{background:var(--accent);border-color:var(--accent);color:#fff}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:640px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--head);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0}
th[data-num],td[data-num]{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:var(--hover)}
tbody tr:last-child td{border-bottom:0}
.r1{color:var(--gold);font-weight:700}
a{color:var(--accent)}
footer{margin-top:30px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:18px}
code{background:var(--head);padding:1px 5px;border-radius:4px;font-size:12.5px}
</style></head><body><div class="wrap">
<h1>Touchstone Benchmark</h1>
<p class="sub">A large-scale benchmark for medical image segmentation. AI algorithms
are trained on AbdomenAtlas and independently evaluated on out-of-distribution CT
from hospitals the models never saw during training.</p>
<ul class="facts">
  <li><b>""" + f"{f['train_volumes']:,}" + """</b>training CT · """ + str(f["train_hospitals"]) + """ hospitals</li>
  <li><b>""" + f"{f['test_volumes']:,}" + """</b>test CT · """ + str(f["test_hospitals"]) + """ hospitals</li>
  <li><b>""" + str(f["organs"]) + """</b>anatomical structures</li>
  <li><b>""" + str(f["algorithms_ranked"]) + """</b>AI algorithms ranked (of """ + str(f["algorithms"]) + """ benchmarked)</li>
  <li><b>""" + str(f["teams"]) + """</b>research teams</li>
</ul>
<div class="tabs" id="tabs" role="tablist"></div>
<div class="scroll"><table id="tbl"><thead></thead><tbody></tbody></table></div>
<footer>
<p>Metric is Dice Similarity Coefficient (DSC), higher is better. Machine-readable
copies of this table live in <code>leaderboard/data/</code> as JSON and CSV.</p>
<p>To add a model, see <a href="https://github.com/MrGiovanni/Touchstone/blob/main/leaderboard/SUBMISSION.md">SUBMISSION.md</a>.
Source: <a href="https://arxiv.org/abs/2411.03670">Touchstone Benchmark (NeurIPS 2024)</a>.</p>
</footer></div>
<script>
const D = """ + data + """;
const tabs=document.getElementById('tabs'), tbl=document.getElementById('tbl');
const views=['Average'].concat(D.organs);
let view='Average', sortKey='value', sortDir=-1;
const meta=Object.fromEntries(D.models.map(m=>[m.model,m]));

function rows(){
  return D.results.filter(r=>r.organ===view).map(r=>{
    const m=meta[r.model]||{};
    return {model:r.model,organization:m.organization||'',value:r.value,
            parameters:m.parameters||'',speed:m.inference_speed||'',
            paper:m.paper||'',code:m.code||''};
  }).sort((a,b)=>{
    const x=a[sortKey],y=b[sortKey];
    if(typeof x==='number'&&typeof y==='number') return (x-y)*sortDir;
    return String(x).localeCompare(String(y))*sortDir;
  });
}
const COLS=[['#',null],['model','model'],['organization','organization'],
            ['DSC','value'],['params','parameters'],['speed','speed'],
            ['paper',null],['code',null]];
function draw(){
  tbl.tHead.innerHTML='<tr>'+COLS.map(([label,key])=>
    `<th ${key==='value'||key==='parameters'?'data-num':''} data-key="${key||''}">${label}</th>`).join('')+'</tr>';
  tbl.tHead.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.key; if(!k) return;
    if(sortKey===k) sortDir*=-1; else {sortKey=k; sortDir=(k==='value')?-1:1;}
    draw();
  });
  tbl.tBodies[0].innerHTML=rows().map((r,i)=>`<tr>
    <td class="${i===0?'r1':''}">${i+1}</td>
    <td class="${i===0?'r1':''}">${r.model}</td>
    <td>${r.organization}</td>
    <td data-num>${r.value.toFixed(1)}</td>
    <td data-num>${r.parameters}</td>
    <td>${r.speed}</td>
    <td>${r.paper?`<a href="${r.paper}" rel="noopener">paper</a>`:''}</td>
    <td>${r.code?`<a href="${r.code}" rel="noopener">code</a>`:''}</td></tr>`).join('');
}
tabs.innerHTML=views.map(v=>
  `<button role="tab" aria-selected="${v===view}" data-v="${v}">${v==='Average'?'Overall':v}</button>`).join('');
tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{
  view=b.dataset.v; sortKey='value'; sortDir=-1;
  tabs.querySelectorAll('button').forEach(x=>x.setAttribute('aria-selected',x===b));
  draw();
});
draw();
</script></body></html>
"""


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "../README.md")
