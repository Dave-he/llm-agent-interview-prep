"use strict";
const $ = (s, p) => (p || document).querySelector(s);
const $$ = (s, p) => Array.from((p || document).querySelectorAll(s));
const esc = s => String(s).replace(/[&<>"']/g, m => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
const REPO = "https://github.com/Dave-he/llm-agent-interview-prep";
const store = {
  get(k, d){ try { const v = localStorage.getItem("ip_" + k); return v ? JSON.parse(v) : d; } catch(e){ return d; } },
  set(k, v){ try { localStorage.setItem("ip_" + k, JSON.stringify(v)); } catch(e){} }
};

const DECK = EXAM20.map(e => ({ key:"ex"+e.n, kind:"考点", front:e.t, sub:"考点 "+e.n+(e.hot?" · 高频":"")+" · 自述骨架后翻面", src:e.s, back:e.p }))
  .concat(FORMULAS.map(f => ({ key:f.id, kind:"公式", front:f.name, sub:"公式速查 · 默写公式+推导+锚点数字", formula:f.f, back:f.back })));

let rate = store.get("rate", {});
let qz = store.get("quiz", { seen:0, correct:0, best:0, wrong:[] });
let plan = store.get("plan", {});
{
  const t = new Date().toISOString().slice(0, 10);
  const days = store.get("days", []);
  if (!days.includes(t)) { days.push(t); store.set("days", days); }
}

const card = { filter:"all", list:[], idx:0, flipped:false };
const quiz = { mode:"all", list:[], idx:0, streak:0, bestR:0, right:0, answered:false };
let designSec = "scene";

function go(view){
  $$("#nav button").forEach(b => b.classList.toggle("on", b.dataset.view === view));
  $$(".view").forEach(s => s.classList.toggle("on", s.id === "v-" + view));
  if (view === "dash") renderDash();
  else if (view === "tree") renderTree();
  else if (view === "cards") renderCards();
  else if (view === "quiz") renderQuiz();
  else if (view === "design") renderDesign();
  else if (view === "plan") renderPlan();
  window.scrollTo(0, 0);
}

function renderDash(){
  const v = $("#v-dash");
  const know = DECK.filter(c => rate[c.key] === "k").length;
  const weak = DECK.filter(c => rate[c.key] === "f" || rate[c.key] === "n").length;
  const seen = qz.seen, acc = seen ? Math.round(qz.correct / seen * 100) : 0;
  const ptotal = PLAN.length * 2;
  const pdone = PLAN.reduce((s, _, i) => s + (plan[i+"t"]?1:0) + (plan[i+"o"]?1:0), 0);
  const dnum = store.get("days", []).length;
  const stat = (cls, val, label) => '<div class="stat ' + cls + '"><b>' + val + '</b><i>' + label + '</i></div>';
  let sug;
  if (weak > 0) sug = '闪卡还有 <b>' + weak + '</b> 张「模糊/不会」——优先清卡，模糊卡比新卡更值得刷。';
  else if ((qz.wrong || []).length > 0) sug = '测验错题池还有 <b>' + qz.wrong.length + '</b> 题，去挑战测验开「只刷错题」。';
  else if (pdone < ptotal) sug = '复习计划还剩 <b>' + (ptotal - pdone) + '</b> 项未打卡，按 7 天微计划推进。';
  else if (!seen) sug = '先来一轮「随机 20 题」测验摸底，再决定背什么。';
  else sug = '全绿。进入模考节奏：抽 3 道场景题，每题限时 15 分钟走完三段式。';
  v.innerHTML = '<h2 class="sec">总览<small>每天 10 分钟，先看建议再动手</small></h2>'
    + '<div class="grid g2"><div class="panel">'
    + '<div class="grid" style="grid-template-columns:repeat(2,1fr);gap:12px">'
    + stat(know / DECK.length >= .8 ? "ok" : know / DECK.length >= .4 ? "mid" : "no", know + "/" + DECK.length, "闪卡已掌握（认识）")
    + stat(seen ? (acc >= 80 ? "ok" : acc >= 60 ? "mid" : "no") : "blue", seen ? acc + "%" : "—", "测验正确率（已答 " + seen + " 题）")
    + stat(pdone / ptotal >= 1 ? "ok" : pdone / ptotal >= .5 ? "mid" : "no", pdone + "/" + ptotal, "7 天计划打卡")
    + stat("gold", dnum, "累计学习天数")
    + '</div>'
    + '<div class="bar"><i style="width:' + Math.round(know / DECK.length * 100) + '%"></i></div>'
    + '<div class="mut">闪卡总进度 ' + know + '/' + DECK.length + '（考点 20 + 公式 8），绿色=≥80%、黄=≥40%</div>'
    + '</div>'
    + '<div class="panel"><h2 class="sec" style="font-size:15px;margin-top:0">今日建议</h2>'
    + '<div class="suggest">' + sug + '</div>'
    + '<div style="margin-top:16px" class="mut">临场表达三动作（面试前夜再看一遍）</div>'
    + '<ul style="margin:8px 0 0 18px;font-size:13.5px;line-height:2">'
    + '<li><b>听题</b>：3 秒归类四型（概念 / 计算 / 设计 / 经验），一句话复述确认再答</li>'
    + '<li><b>卡壳</b>：降级到万能四段——锚点 → 机制 → 数字 → 代价</li>'
    + '<li><b>表达</b>：先结论，每层 15–30 秒，主动交棒「这块能展开，看您想深入哪」</li>'
    + '</ul></div></div>'
    + '<div class="panel" style="margin-top:14px"><h2 class="sec" style="font-size:15px;margin-top:0">深读原文</h2><div class="mut">'
    + '<a target="_blank" rel="noopener" href="' + REPO + '/blob/main/备考总纲-LLM-Agent.md">备考总纲</a> · '
    + '<a target="_blank" rel="noopener" href="' + REPO + '/blob/main/Agent搭建面试专项.md">Agent 搭建专项</a> · '
    + '<a target="_blank" rel="noopener" href="' + REPO + '">GitHub 仓库</a></div></div>';
}

function renderTree(){
  const v = $("#v-tree");
  v.innerHTML = '<h2 class="sec">知识树<small>三层十八节点 · 点击节点展开背诵骨架</small></h2>'
    + '<input type="search" id="treeq" placeholder="搜关键词，如 GQA / 注入 / compaction / DPO …">'
    + '<div id="treewrap" style="margin-top:14px"></div>';
  const draw = () => {
    const kw = $("#treeq").value.trim().toLowerCase();
    $("#treewrap").innerHTML = TREE.map((L, li) => {
      const kids = L.kids.filter(k => !kw || (k.t + k.d).toLowerCase().includes(kw));
      if (kw && !kids.length) return "";
      return '<div class="layer' + (kw || li === 0 ? " open" : "") + '"><h3>' + esc(L.name) + '<em>' + esc(L.note) + '</em></h3><div class="kids">'
        + kids.map(k => '<div class="knode"><b>' + esc(k.t) + '</b><p>' + esc(k.d) + '</p></div>').join("")
        + '</div></div>';
    }).join("") || '<div class="mut">无匹配节点</div>';
    $$(".layer>h3", v).forEach(h => h.onclick = () => h.parentElement.classList.toggle("open"));
    $$(".knode", v).forEach(n => n.onclick = () => n.classList.toggle("open"));
  };
  $("#treeq").oninput = draw;
  draw();
}

function shuffle(a){ for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; }

function deckFor(f){
  if (f === "exam") return DECK.filter(c => c.kind === "考点");
  if (f === "formula") return DECK.filter(c => c.kind === "公式");
  if (f === "weak") return DECK.filter(c => rate[c.key] === "f" || rate[c.key] === "n");
  return DECK.slice();
}

function renderCards(){
  const v = $("#v-cards");
  if (!card.list.length && card.filter !== "weak") card.list = deckFor(card.filter);
  const list = card.list;
  const c = list[card.idx];
  const know = list.filter(x => rate[x.key] === "k").length;
  v.innerHTML = '<h2 class="sec">闪卡速查<small>考点 20 + 公式 8 · 先自述再翻面对答案</small></h2>'
    + '<div class="chips">'
    + [["all","全部"],["exam","考点 20"],["formula","公式 8"],["weak","模糊+不会"]].map(x =>
        '<button class="chip' + (card.filter === x[0] ? " on" : "") + '" data-f="' + x[0] + '">' + x[1] + '</button>').join("")
    + '<button class="chip" data-f="shuf">打乱顺序</button>'
    + '</div>'
    + (list.length && c
      ? '<div class="mut" style="margin-bottom:8px">本组进度：认识 ' + know + ' / ' + list.length
        + (rate[c.key] ? ' · 本卡上次标记：<b>' + ({k:"认识",f:"模糊",n:"不会"}[rate[c.key]]) + '</b>' : '')
        + '</div>'
        + '<div class="flipwrap"><div class="flip' + (card.flipped ? " go" : "") + '" id="flip">'
        + '<div class="face"><div class="q">' + esc(c.front) + '</div><div class="tip">' + esc(c.sub) + '</div></div>'
        + '<div class="face back"><h4>' + esc(c.front) + '</h4>'
        + (c.formula ? '<pre>' + esc(c.formula) + '</pre>' : "")
        + '<div style="line-height:1.9;text-align:left">' + esc(c.back) + '</div>'
        + (c.src ? '<div class="tip">来源：' + esc(c.src) + '</div>' : "")
        + '</div></div></div>'
        + '<div class="navrow">'
        + '<div><button class="btn mini" data-act="prev">上一张</button> <button class="btn mini" data-act="flipbtn">翻面</button></div>'
        + '<div class="counter">' + (card.idx + 1) + " / " + list.length + '</div></div>'
        + '<div class="chips" style="margin-top:4px">'
        + '<button class="btn rate-know" data-act="k">认识</button>'
        + '<button class="btn rate-fuzzy" data-act="f">模糊</button>'
        + '<button class="btn rate-no" data-act="n">不会</button>'
        + '</div>'
      : '<div class="panel mut">本组没有卡片——「模糊+不会」清空即闪卡全掌握，或切回「全部」。' + (card.filter === "weak" ? ' 目前 0 张薄弱卡。' : '') + '</div>');
  const flip = $("#flip");
  if (flip) flip.onclick = () => { card.flipped = !card.flipped; flip.classList.toggle("go", card.flipped); };
  $$(".chip", v).forEach(b => b.onclick = () => {
    const f = b.dataset.f;
    card.flipped = false;
    if (f === "shuf") { card.list = shuffle(list.slice()); card.idx = 0; }
    else { card.filter = f; card.list = deckFor(f); card.idx = 0; }
    renderCards();
  });
  $$(".btn[data-act]", v).forEach(b => b.onclick = () => {
    const a = b.dataset.act;
    if (a === "prev") { card.idx = (card.idx - 1 + list.length) % list.length; card.flipped = false; }
    else if (a === "flipbtn") { card.flipped = !card.flipped; }
    else { rate[c.key] = a; store.set("rate", rate); card.idx = (card.idx + 1) % list.length; card.flipped = false; }
    renderCards();
  });
}
function renderQuiz(){
  const v = $("#v-quiz");
  if (quiz.list.length && quiz.idx < quiz.list.length) return quizQ(v);
  if (quiz.list.length) return quizSum(v);
  quizHome(v);
}

function quizHome(v){
  const wl = (qz.wrong || []).length;
  v.innerHTML = '<h2 class="sec">挑战测验<small>' + QUIZ.length + ' 题 · 键盘 1–4 作答、Enter 下一题</small></h2>'
    + '<div class="grid g3">'
    + '<div class="stat blue"><b>' + qz.correct + "/" + qz.seen + '</b><i>累计正确 / 已答' + (qz.seen ? "（正确率 " + Math.round(qz.correct / qz.seen * 100) + "%）" : "") + '</i></div>'
    + '<div class="stat gold"><b>' + qz.best + '</b><i>最佳连对</i></div>'
    + '<div class="stat ' + (wl ? "no" : "ok") + '"><b>' + wl + '</b><i>错题池</i></div>'
    + '</div>'
    + '<div class="grid g3" style="margin-top:16px">'
    + '<button class="btn pri" data-m="all">全量 ' + QUIZ.length + ' 题</button>'
    + '<button class="btn" data-m="rand">随机 20 题</button>'
    + '<button class="btn" data-m="wrong">只刷错题（' + wl + '）</button>'
    + '</div>'
    + '<div class="mut" style="margin-top:12px">建议节奏：先随机 20 摸底 → 错题清池 → 全量过一遍巩固。</div>';
  $$("button[data-m]", v).forEach(b => b.onclick = () => {
    quiz.mode = b.dataset.m;
    quiz.list = quiz.mode === "all" ? QUIZ.map((_, i) => i)
      : quiz.mode === "rand" ? shuffle(QUIZ.map((_, i) => i)).slice(0, 20)
      : shuffle((qz.wrong || []).slice());
    quiz.idx = 0; quiz.streak = 0; quiz.bestR = 0; quiz.right = 0; quiz.answered = false;
    renderQuiz();
  });
}

function quizQ(v){
  const qi = quiz.list[quiz.idx];
  const q = QUIZ[qi];
  const modeName = quiz.mode === "wrong" ? "错题重刷" : quiz.mode === "rand" ? "随机 20" : "全量 " + QUIZ.length;
  v.innerHTML = '<div class="qwrap">'
    + '<div class="qmeta"><span>第 ' + (quiz.idx + 1) + ' / ' + quiz.list.length + ' 题 · ' + modeName + '</span>'
    + '<span>连对 <span class="streak">' + quiz.streak + '</span></span></div>'
    + '<div class="qq">' + esc(q.q) + '</div>'
    + q.o.map((o, i) => '<button class="opt" data-i="' + i + '"><b style="color:var(--sub);margin-right:8px">' + (i + 1) + '</b>' + esc(o) + '</button>').join("")
    + '<div class="expl" id="expl"></div>'
    + '<div class="navrow"><button class="btn pri" id="qnext" style="display:none">下一题</button>'
    + '<button class="btn mini" id="qquit">结束本轮</button></div>'
    + '</div>';
  const opts = $$(".opt", v);
  const next = () => { quiz.idx++; quiz.answered = false; renderQuiz(); };
  const answer = i => {
    if (quiz.answered) return;
    quiz.answered = true;
    const ok = i === q.a;
    opts.forEach((b, j) => { b.disabled = true; if (j === q.a) b.classList.add("right"); if (j === i && !ok) b.classList.add("wrong"); });
    qz.seen++;
    if (ok) { qz.correct++; quiz.streak++; quiz.right++; quiz.bestR = Math.max(quiz.bestR, quiz.streak); qz.best = Math.max(qz.best, quiz.streak); }
    else quiz.streak = 0;
    qz.wrong = (qz.wrong || []).filter(x => x !== qi);
    if (!ok) qz.wrong.push(qi);
    store.set("quiz", qz);
    const ex = $("#expl", v);
    ex.classList.add("show");
    ex.innerHTML = (ok ? '<b style="color:var(--acc)">答对</b> · ' : '<b style="color:var(--bad)">答错</b> · 正确答案：' + esc(q.o[q.a]) + '<br>') + esc(q.e);
    const nx = $("#qnext", v);
    nx.style.display = "";
    nx.textContent = quiz.idx + 1 >= quiz.list.length ? "查看结果" : "下一题";
    nx.onclick = next;
    nx.focus();
  };
  opts.forEach((b, i) => b.onclick = () => answer(i));
  $("#qquit", v).onclick = () => { quiz.idx = quiz.list.length; renderQuiz(); };
}

function quizSum(v){
  const n = quiz.list.length;
  if (!n) { quiz.list = []; return renderQuiz(); }
  const r = quiz.right, pct = Math.round(r / n * 100);
  v.innerHTML = '<h2 class="sec">本轮结果</h2><div class="qwrap" style="text-align:center">'
    + '<div style="font-size:46px;font-weight:700;color:' + (pct >= 80 ? "var(--acc)" : pct >= 60 ? "var(--warn)" : "var(--bad)") + '">' + pct + '%</div>'
    + '<div class="mut" style="margin:8px 0 16px">' + r + ' / ' + n + ' 正确 · 本轮最佳连对 ' + quiz.bestR + ' · 错题池现有 ' + (qz.wrong || []).length + ' 题</div>'
    + '<div class="chips" style="justify-content:center">'
    + '<button class="btn pri" id="qagain">再来一轮</button>'
    + '<button class="btn" id="qwrong">只刷错题</button>'
    + '<button class="btn" id="qhome">返回首页</button>'
    + '</div></div>';
  $("#qagain", v).onclick = () => { quiz.list = []; renderQuiz(); const b = $('button[data-m="' + quiz.mode + '"]', v); if (b) b.click(); };
  $("#qwrong", v).onclick = () => { quiz.list = []; renderQuiz(); const b = $('button[data-m="wrong"]', v); if (b) b.click(); };
  $("#qhome", v).onclick = () => { quiz.list = []; renderQuiz(); };
}

function renderDesign(){
  const v = $("#v-design");
  const tabs = [["scene","场景设计题 P1–P6"],["code","手撕代码 T1–T6"],["qa","速答卡 Top30"]];
  let html = '<h2 class="sec">场景 · 手撕<small>点击卡片展开 · 先自己想 30 秒再看答案</small></h2><div class="chips">'
    + tabs.map(x => '<button class="chip' + (designSec === x[0] ? " on" : "") + '" data-s="' + x[0] + '">' + x[1] + '</button>').join("") + '</div>';
  if (designSec === "scene") {
    html += DESIGN.map(d => '<div class="dsg"><h3><span>' + esc(d.id + " · " + d.t) + '</span><span class="tag">' + esc(d.tag) + '</span></h3><div class="bd">'
      + '<div class="mut" style="margin:10px 0">' + esc(d.brief) + '</div>'
      + '<h5>3 分钟版答题骨架</h5><ul>' + d.min3.map(x => "<li>" + esc(x) + "</li>").join("") + '</ul>'
      + '<h5>追问树</h5><ul>' + d.drill.map(x => "<li>" + esc(x) + "</li>").join("") + '</ul>'
      + '<div class="trap">陷阱：' + esc(d.trap) + '</div></div></div>').join("");
  } else if (designSec === "code") {
    html += '<div class="mut" style="margin-bottom:10px">面试手撕顺序建议 T1→T2→T4→T5→T3→T6；每段先说思路再看代码。</div>'
      + CODE.map(c => '<div class="excard"><h4><span>' + esc(c.id + " · " + c.t) + '</span></h4><div class="body">'
      + '<p style="margin:4px 0"><b>思路：</b>' + esc(c.idea) + '</p>'
      + '<pre>' + esc(c.code) + '</pre>'
      + '<div class="trap">' + esc(c.pit) + '</div></div></div>').join("");
  } else {
    html += '<div class="mut" style="margin-bottom:10px">遮住答案先自己说一句再点开对照——目标 10 秒内开口，一句到点。</div>'
      + QA30.map(x => '<div class="excard"><h4><span>' + esc(x.q) + '</span><span class="pill">' + esc(x.c) + '</span></h4><div class="body"><b>' + esc(x.a) + '</b></div></div>').join("");
  }
  v.innerHTML = html;
  $$(".chip", v).forEach(b => b.onclick = () => { designSec = b.dataset.s; renderDesign(); });
  $$(".dsg>h3", v).forEach(h => h.onclick = () => h.parentElement.classList.toggle("open"));
  $$(".excard", v).forEach(c => c.onclick = () => c.classList.toggle("open"));
}

function renderPlan(){
  const v = $("#v-plan");
  const ptotal = PLAN.length * 2;
  const pdone = PLAN.reduce((s, _, i) => s + (plan[i+"t"]?1:0) + (plan[i+"o"]?1:0), 0);
  v.innerHTML = '<h2 class="sec">复习计划<small>专项 7 天微计划 · 任务与产出双打卡</small></h2>'
    + '<div class="panel"><div class="bar"><i style="width:' + Math.round(pdone / ptotal * 100) + '%"></i></div>'
    + '<div class="mut">' + pdone + ' / ' + ptotal + ' 项完成' + (pdone === ptotal ? ' —— 计划通关，进模考节奏。' : '') + '</div></div>'
    + '<div style="margin-top:14px">'
    + PLAN.map((p, i) => {
      const dt = plan[i+"t"], doo = plan[i+"o"];
      return '<div class="pday' + (dt && doo ? " done" : "") + '">'
        + '<div style="display:flex;gap:10px;align-items:flex-start;flex:1">'
        + '<input type="checkbox" data-k="' + i + 't"' + (dt ? " checked" : "") + '>'
        + '<div style="flex:1"><b>' + esc(p.d) + ' · ' + esc(p.t) + '</b><span>产出要求：' + esc(p.o) + '</span></div></div>'
        + '<div class="mut" style="display:flex;align-items:center;gap:6px">产出<input type="checkbox" data-k="' + i + 'o"' + (doo ? " checked" : "") + '></div>'
        + '</div>';
    }).join("")
    + '</div><div style="margin-top:14px"><button class="btn mini" id="preset">重置打卡</button></div>';
  $$("input[type=checkbox]", v).forEach(cb => cb.onchange = () => { plan[cb.dataset.k] = cb.checked; store.set("plan", plan); renderPlan(); });
  $("#preset", v).onclick = () => { if (confirm("重置全部打卡记录？")) { plan = {}; store.set("plan", plan); renderPlan(); } };
}

document.addEventListener("keydown", e => {
  if (!$("#v-quiz") || !$("#v-quiz").classList.contains("on")) return;
  const opts = $$("#v-quiz .opt");
  if (!opts.length) return;
  if (!quiz.answered) {
    if (e.key >= "1" && e.key <= "4") { const i = +e.key - 1; if (i < opts.length) opts[i].click(); }
  } else if (e.key === "Enter") {
    const nx = $("#qnext");
    if (nx) { e.preventDefault(); nx.click(); }
  }
});

$$("#nav button").forEach(b => b.onclick = () => go(b.dataset.view));
go("dash");
