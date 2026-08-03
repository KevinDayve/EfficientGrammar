"""Local web demo for EmailGrammar.

    python serve.py            # then open http://localhost:8000

Three tiers, live: spelling auto-fixes (applied), spelling SUGGESTIONS (click a
red-underlined word -> pick from candidates), and grammar suggestions (click Apply).
Loads the model once at startup.
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from emailgrammar.pipeline import build_pipeline

app = FastAPI(title="EmailGrammar demo")
PIPE = build_pipeline(model="mini", beam_size=2)

_SENT = re.compile(r"[^.!?]+[.!?]*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def diff_tokens(before: str, after: str) -> list[dict]:
    a, b = before.split(), after.split()
    sm = SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b])
    out: list[dict] = []
    for op, _i1, _i2, j1, j2 in sm.get_opcodes():
        for w in b[j1:j2]:
            out.append({"t": w, "c": op != "equal"})
    return out


class Req(BaseModel):
    text: str


@app.post("/analyze")
def analyze(req: Req):
    t0 = time.perf_counter()
    tokens = PIPE.speller.analyze(req.text) if PIPE.speller else []
    grammar = []
    sents = split_sentences(req.text)
    if sents:
        for raw, c in zip(sents, PIPE.correct_batch(sents, detailed=True, max_batch_size=16)):
            if c.final.strip() != c.spell_corrected.strip():
                grammar.append({
                    "raw": raw,
                    "final": c.final,
                    "diff": diff_tokens(c.spell_corrected, c.final),
                    "guarded": c.meaning_guarded,
                })
    return {"latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "tokens": tokens, "grammar": grammar}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EmailGrammar — live demo</title>
<style>
  :root{--bg:#f6f7f9;--card:#fff;--ink:#111;--muted:#6b7280;--line:#e5e7eb;
        --green:#dcfce7;--greeni:#166534;--blue:#dbeafe;--bluei:#1e40af;--accent:#2a78d6;--red:#dc2626;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
  header{padding:22px 28px;border-bottom:1px solid var(--line);background:var(--card)}
  h1{margin:0;font-size:19px} .sub{color:var(--muted);font-size:13px;margin-top:3px}
  .wrap{max-width:1120px;margin:22px auto;padding:0 20px;display:grid;
        grid-template-columns:1fr 1fr;gap:20px}
  @media(max-width:840px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
  .lbl{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
       letter-spacing:.04em;margin-bottom:10px}
  textarea{width:100%;min-height:250px;border:none;outline:none;resize:vertical;
           font:15px/1.7 system-ui,sans-serif;color:var(--ink)}
  #review{min-height:120px;white-space:pre-wrap;line-height:1.9}
  .fix{background:var(--green);color:var(--greeni);border-radius:4px;padding:0 3px}
  .sug{text-decoration:underline wavy var(--red);text-underline-offset:3px;cursor:pointer}
  .sug:hover{background:#fee2e2}
  .unk{text-decoration:underline dotted var(--muted);text-underline-offset:3px}
  .menu{position:absolute;z-index:50;background:#fff;border:1px solid var(--line);
        border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.13);padding:6px;min-width:150px}
  .menu .h{font-size:11px;color:var(--muted);padding:4px 8px}
  .menu button{display:block;width:100%;text-align:left;border:none;background:none;
        padding:7px 9px;border-radius:6px;font:14px system-ui;cursor:pointer;color:var(--ink)}
  .menu button:hover{background:#eff6ff;color:var(--accent)}
  .sugcard{border-top:1px solid var(--line);padding:11px 0;display:flex;gap:12px;
           align-items:flex-start;justify-content:space-between}
  .sugcard:first-child{border-top:none}
  mark.gr{background:var(--blue);color:var(--bluei);border-radius:4px;padding:0 2px}
  .apply{border:1px solid var(--accent);color:var(--accent);background:#fff;border-radius:7px;
         padding:5px 12px;font:13px system-ui;cursor:pointer;white-space:nowrap;flex-shrink:0}
  .apply:hover{background:var(--accent);color:#fff}
  .guard{color:var(--muted);font-size:12px;font-style:italic}
  .none{color:var(--muted);font-style:italic}
  .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:3px;vertical-align:middle;margin-right:5px}
  .foot{max-width:1120px;margin:6px auto 30px;padding:0 20px;color:var(--muted);font-size:12px}
  .pill{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:20px;padding:2px 9px}
</style></head><body>
<header>
  <h1>EmailGrammar <span class="pill">CPU-only · no LLM</span></h1>
  <div class="sub">Obvious typos auto-fix. Unsure ones become <b>clickable suggestions</b>. Grammar is offered, never forced.</div>
</header>
<div class="wrap">
  <div class="card">
    <div class="lbl">Type an email</div>
    <textarea id="in" spellcheck="false"></textarea>
  </div>
  <div class="card">
    <div class="lbl">Review — click a red word to pick a spelling</div>
    <div id="review"></div>
    <div class="lbl" style="margin-top:18px">Grammar — suggestions</div>
    <div id="gram"></div>
    <div class="legend">
      <span><span class="dot" style="background:var(--green)"></span>auto-fixed</span>
      <span><span class="dot" style="background:#fecaca"></span>spelling suggestion (click)</span>
      <span><span class="dot" style="background:var(--blue)"></span>grammar suggestion</span>
    </div>
  </div>
</div>
<div class="foot"><span id="stat"></span></div>
<div id="menu" class="menu" style="display:none"></div>
<script>
const $=s=>document.querySelector(s);
const DEMO="Hey Kevin, i recieved teh msg last nite. I was kude to you. Their are a few things we shud dicuss. Dont forget to email legal@rediff.com by 5pm.";
let TOKENS=[];

function esc(t){const d=document.createElement('span');d.textContent=t;return d;}

function renderReview(tokens){
  const el=$('#review'); el.textContent='';
  tokens.forEach((tk,idx)=>{
    if(tk.kind==='space'){el.appendChild(document.createTextNode(tk.raw));return;}
    if(tk.pre) el.appendChild(document.createTextNode(tk.pre));
    let w;
    if(tk.status==='autofix'){w=document.createElement('span');w.className='fix';
      w.textContent=tk.fix;w.title='auto-corrected from "'+tk.core+'"';}
    else if(tk.status==='suggest'){w=document.createElement('span');w.className='sug';
      w.textContent=tk.core;w.onclick=e=>openMenu(e,idx,tk.candidates||[]);}
    else if(tk.status==='unknown'){w=document.createElement('span');w.className='unk';w.textContent=tk.core;}
    else {w=esc(tk.core);}
    el.appendChild(w);
    if(tk.post) el.appendChild(document.createTextNode(tk.post));
  });
}

function openMenu(ev,idx,cands){
  ev.stopPropagation();
  const m=$('#menu'); m.textContent='';
  const h=document.createElement('div');h.className='h';h.textContent=cands.length?'Replace with…':'No suggestions';m.appendChild(h);
  cands.forEach(c=>{const b=document.createElement('button');b.textContent=c;
    b.onclick=()=>applySpelling(idx,c);m.appendChild(b);});
  const r=ev.target.getBoundingClientRect();
  m.style.left=(window.scrollX+r.left)+'px';
  m.style.top=(window.scrollY+r.bottom+4)+'px';
  m.style.display='block';
}
function closeMenu(){$('#menu').style.display='none';}
document.addEventListener('click',closeMenu);

function applySpelling(idx,word){
  // rebuild the raw text, replacing ONLY the clicked word (keep the rest as typed)
  const text=TOKENS.map((tk,i)=>{
    if(tk.kind==='space') return tk.raw;
    return tk.pre + (i===idx?word:tk.core) + tk.post;
  }).join('');
  $('#in').value=text; closeMenu(); run();
}
function applyGrammar(raw,final){
  $('#in').value=$('#in').value.replace(raw,final); run();
}

function grTokens(list){const f=document.createDocumentFragment();
  list.forEach((tk,i)=>{if(i)f.appendChild(document.createTextNode(' '));
    const s=document.createElement(tk.c?'mark':'span');if(tk.c)s.className='gr';
    s.textContent=tk.t;f.appendChild(s);});return f;}

async function run(){
  const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:$('#in').value})});
  const d=await r.json();
  TOKENS=d.tokens; renderReview(d.tokens);
  const g=$('#gram'); g.textContent='';
  if(!d.grammar.length){g.innerHTML='<span class="none">No grammar suggestions — looks good.</span>';}
  d.grammar.forEach(s=>{
    const row=document.createElement('div');row.className='sugcard';
    const left=document.createElement('div');left.appendChild(grTokens(s.diff));
    if(s.guarded){const n=document.createElement('div');n.className='guard';
      n.textContent='kept as-is to preserve meaning';left.appendChild(n);}
    row.appendChild(left);
    if(!s.guarded){const b=document.createElement('button');b.className='apply';b.textContent='Apply';
      b.onclick=()=>applyGrammar(s.raw,s.final);row.appendChild(b);}
    g.appendChild(row);
  });
  $('#stat').textContent=(d.tokens.filter(t=>t.status==='suggest').length)+' spelling suggestion(s) · '
    +d.grammar.length+' grammar · '+d.latency_ms+' ms';
}
let t;$('#in').addEventListener('input',()=>{clearTimeout(t);t=setTimeout(run,350);});
$('#in').value=DEMO; run();
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
