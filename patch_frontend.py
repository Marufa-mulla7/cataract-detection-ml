"""
patch_frontend.py - updates ocuvision-concept.html with the FINAL results and fixes.

Run from the project folder (the one containing ocuvision-concept.html):

    python patch_frontend.py

It first saves a backup (ocuvision-concept.backup.html), then applies every
change to a copy of that backup and prints which ones were applied. It is safe
to run more than once. Nothing else in your file is touched.
"""
import os
import re
import shutil

FILE = "ocuvision-concept.html"
BACKUP = "ocuvision-concept.backup.html"

if not os.path.exists(FILE):
    raise SystemExit(f"Cannot find {FILE}. Run this script from the project folder.")
if not os.path.exists(BACKUP):
    shutil.copyfile(FILE, BACKUP)
    print(f"Backup saved as {BACKUP}")

# Always patch from the untouched backup, so running this script twice is safe.
with open(BACKUP, encoding="utf-8") as fh:
    text = fh.read()

applied, missed = 0, []


def rep(old, new, label):
    """Plain text replacement."""
    global text, applied
    if old in text:
        text = text.replace(old, new)
        applied += 1
    else:
        missed.append(label)


def rx(pattern, new, label):
    """Regex replacement (new text is inserted literally)."""
    global text, applied
    text, n = re.subn(pattern, lambda m: new, text, flags=re.S)
    if n:
        applied += 1
    else:
        missed.append(label)


# ---------------------------------------------------------------- branding
rep("<title>Eye Anomaly Detection — Concept Preview</title>",
    "<title>Eye Anomaly Detection — AI Screening</title>", "title")
rep('<div class="nav-item" data-page="analyze">📤 Upload Scan</div>', "", "duplicate nav item")

# ---------------------------------------------------------------- dashboard / analytics text
rep("Upload a retinal or ocular image and let the computer-vision model analyze visual patterns, detect abnormalities, and generate an interpretable screening report.",
    "Upload a fundus (retinal) image and let two deep-learning models screen it for five conditions, explain the result with Grad-CAM, and generate a screening report.",
    "hero text")
rep('<div class="float-label fl-2">AI Confidence<b>96.8%</b></div>',
    '<div class="float-label fl-2">ResNet50 Test Accuracy<b>95.2%</b></div>', "hero label")
rep("<td>OCT</td>", "<td>Fundus</td>", "OCT example rows")
rep('<div class="card-eyebrow">Model Accuracy</div><div class="stat-num" style="color:var(--purple)">96.2%</div>',
    '<div class="card-eyebrow">Model Accuracy (ResNet50 test)</div><div class="stat-num" style="color:var(--purple)">95.2%</div>',
    "analytics accuracy")

# ---------------------------------------------------------------- upload page
rep("<h3>Drag &amp; drop a retinal, fundus, OCT, or ocular image</h3>",
    "<h3>Drag &amp; drop a fundus (retinal) image</h3>", "upload heading")
rep('<div class="format-row">JPG • JPEG • PNG • TIFF</div>',
    '<div class="format-row">JPG • JPEG • PNG • TIFF</div>'
    '<p style="font-size:12px; color:var(--amber); margin:-8px 0 16px 0;">Fundus (retinal) photographs only. Academic screening aid, not a medical device.</p>',
    "format note")
rep('<button class="btn btn-ghost">Take Photo</button>', "", "take photo button")
rep('<button class="btn btn-ghost" onclick="useSample()">Use Sample Image</button>', "", "sample button")
rep('<b id="metaRes">1024 × 1024</b>', '<b id="metaRes">—</b>', "resolution placeholder")
rep('<div class="meta-row"><span>Image quality</span><b style="color:var(--green)">Excellent</b></div>',
    '<div class="meta-row"><span>Image quality</span><b>Not assessed</b></div>', "image quality")
rep("document.getElementById('metaSize').textContent = (file.size/1024).toFixed(0)+' KB';",
    "document.getElementById('metaSize').textContent = (file.size/1024).toFixed(0)+' KB';\n"
    "    previewImg.onload = () => { document.getElementById('metaRes').textContent = previewImg.naturalWidth + ' × ' + previewImg.naturalHeight; };",
    "resolution script")
rep("This output is not a definitive medical diagnosis.",
    "This output is not a definitive medical diagnosis, and confidence scores are approximate, not guaranteed probabilities.",
    "results disclaimer")

# ---------------------------------------------------------------- model card: training data
rep('<td class="mono">300</td><td>Cataract Dataset (jr2ngb/cataractdataset)</td>',
    '<td class="mono">900</td><td>847 real (eyedis 598, NL 249) + 53 augmented (training only)</td>', "normal row")
rep('<td class="mono">382</td><td>Cataract Dataset + ODIR-5K boost</td>', '<td class="mono">900</td><td>876 real (eyedis 511, odir 279, cataract 86) + 24 augmented (training only)</td>', "cataract row")
rep('<td class="mono">340</td><td>Cataract Dataset + ODIR-5K boost</td>', '<td class="mono">900</td><td>864 real (eyedis 552, odir 236, Glaucoma 76) + 36 augmented (training only)</td>', "glaucoma row")
rep('<td class="mono">300</td><td>RFMiD (100 real + 200 synthetic augmented)</td>',
    '<td class="mono">900</td><td>188 real sources (RFMiD 100 + ADAM 88) + 712 augmented (training only)</td>', "amd row")
rep('<td class="mono">1,857</td><td>APTOS 2019</td>',
    '<td class="mono">1,000</td><td>APTOS 2019: 971 unique real + 29 augmented (training only)</td>', "dr row")
rx(r'<b style="color:var\(--text\)">3,479 images total</b>.*?476 held-out test images\.',
   '<b style="color:var(--text)">4,600 images total</b>, split into train / validation / test (3,434 / 583 / 583). '
   'Validation and test sets contain real images only; augmented images are in the training split only. '
   'All test-set figures below are computed on the 583 held-out test images.', "dataset totals")
rx(r"⚠ Honest caveat:.*?Known Limitations\.",
   "⚠ Honest caveat: 712 of AMD's 900 images (84% of its training images) are augmented copies of only 132 real training photos "
   "(188 unique real AMD sources overall), and Diabetic Retinopathy is the largest class at 1,000 of 4,600 images (21.7%). "
   "Both facts affect how the per-class results below should be read. See Known Limitations.", "caveat")

# ---------------------------------------------------------------- model card: performance
rep("<b>96.01%</b>", "<b>95.03%</b>", "resnet val acc")
rep("<b>97%</b>", "<b>95.20% (555/583)</b>", "resnet test acc")
rep("<b>0.94</b>", "<b>0.96</b>", "resnet macro f1")
rep("<b>20 (no early stop)</b>", "<b>20 (best at epoch 17)</b>", "resnet epochs")
rep("<b>337.5 min</b>", "<b>about 6 h 56 min (CPU)</b>", "resnet time")
rep("<b>94.33%</b>", "<b>96.23%</b>", "effnet val acc")
rep("<b>96%</b>", "<b>96.23% (561/583)</b>", "effnet test acc")
rep("<b>0.92</b>", "<b>0.96</b>", "effnet macro f1")
rep("<b>10 (early stopped)</b>", "<b>20 (best at epoch 20)</b>", "effnet epochs")
rep("<b>123.3 min</b>", "<b>about 3 h (CPU)</b>", "effnet time")

# per-class F1 (EfficientNet first, then ResNet50)
rep('<td>Normal</td><td class="mono">0.89</td><td class="mono">0.88</td>',
    '<td>Normal</td><td class="mono">0.94</td><td class="mono">0.93</td>', "f1 normal")
rep('<td>Cataract</td><td class="mono">0.90</td><td class="mono">0.92</td>',
    '<td>Cataract</td><td class="mono">0.96</td><td class="mono">0.95</td>', "f1 cataract")
rep('<td>Glaucoma</td><td class="mono">0.84</td><td class="mono">0.89</td>',
    '<td>Glaucoma</td><td class="mono">0.94</td><td class="mono">0.92</td>', "f1 glaucoma")
rep('<td>AMD</td><td class="mono">0.99</td><td class="mono">0.99</td>',
    '<td>AMD</td><td class="mono">0.98</td><td class="mono">0.98</td>', "f1 amd")
rep('<td>Diabetic Retinopathy</td><td class="mono">0.99</td><td class="mono">1.00</td>',
    '<td>Diabetic Retinopathy</td><td class="mono">1.00</td><td class="mono">1.00</td>', "f1 dr")

# ---------------------------------------------------------------- failure cases (from the ResNet50 confusion matrix)
rep('<td>Normal</td><td class="mono">2</td><td class="mono">46</td><td><span class="badge badge-amber">4.3%</span></td>',
    '<td>Normal</td><td class="mono">9</td><td class="mono">135</td><td><span class="badge badge-amber">6.7%</span></td>', "fail normal")
rep('<td>Cataract</td><td class="mono">5</td><td class="mono">52</td><td><span class="badge badge-amber">9.6%</span></td>',
    '<td>Cataract</td><td class="mono">11</td><td class="mono">135</td><td><span class="badge badge-amber">8.1%</span></td>', "fail cataract")
rep('<td>Glaucoma</td><td class="mono">8</td><td class="mono">44</td><td><span class="badge badge-red">18.2%</span></td>',
    '<td>Glaucoma</td><td class="mono">7</td><td class="mono">135</td><td><span class="badge badge-amber">5.2%</span></td>', "fail glaucoma")
rep('<td>AMD</td><td class="mono">0</td><td class="mono">42</td><td><span class="badge badge-green">0%</span></td>',
    '<td>AMD</td><td class="mono">1</td><td class="mono">28</td><td><span class="badge badge-amber">3.6%</span></td>', "fail amd")
rep('<td>Diabetic Retinopathy</td><td class="mono">0</td><td class="mono">292</td>',
    '<td>Diabetic Retinopathy</td><td class="mono">0</td><td class="mono">150</td>', "fail dr")
rx(r"Every misclassification falls within the Normal / Cataract / Glaucoma trio.*?zero test-set errors\.",
   "27 of the 28 errors fall within the Normal / Cataract / Glaucoma trio, and Glaucoma is involved in 23 of them. "
   "The single AMD error was predicted as Glaucoma, and Diabetic Retinopathy has zero test-set errors. "
   "AMD's test set contains only 28 images, so its error rate is imprecise.", "failure text")

# ---------------------------------------------------------------- agreement and cost-weighted cards
rep("Agreement on 476 test images", "Agreement on 583 test images", "agreement heading")
rep("<b>459 / 476 (96.4%)</b>", "<b>552 / 583 (94.7%)</b>", "models agree")
rep('<b style="color:var(--green)">98.04%</b>', '<b style="color:var(--green)">98.37%</b>', "acc when agree")
rep("<b>17 / 476 (3.6%)</b>", "<b>31 / 583 (5.3%)</b>", "models disagree")
rep('<b style="color:var(--amber)">64.71%</b>', '<b style="color:var(--red)">38.71%</b>', "primary when disagree")
rep('<b style="color:var(--red)">29.41%</b>', '<b style="color:var(--green)">58.06%</b>', "second when disagree")
rep("<b>96.8% (461/476)</b>", "<b>95.2% (555/583)</b>", "plain accuracy")
rep("<b>0.9754</b>", "<b>0.9744</b>", "cost score")
rep('<b style="color:var(--red)">10 cases</b>', '<b style="color:var(--red)">9 cases</b>', "missed disease")
rx(r"(False alarm \(Normal → disease\)</span><b>)2 cases", "False alarm (Normal → disease)</span><b>9 cases", "false alarm")

# ---------------------------------------------------------------- limitations
rx(r'<li><b style="color:var\(--text\)">AMD\'s training data is 67% synthetic</b>.*?</li>',
   '<li><b style="color:var(--text)">AMD\'s training data is 84% augmented</b> (712 of 844 training images are augmented copies of only 132 real training photos; 188 unique real sources overall), '
   'so its near-perfect F1 may partly reflect recognizing "known source image, transformed" rather than fully generalizable AMD features. Its test set has only 28 images.</li>',
   "limitation amd")
rx(r'<li><b style="color:var\(--text\)">Small per-bin sample sizes</b>.*?</li>',
   '<li><b style="color:var(--text)">Confidence is only approximately calibrated.</b> The expected calibration error is 0.031 for both models, but some errors carry high confidence. The original Monte Carlo Dropout threshold (0.15) caught only 1 of 15 validation errors. '
   'The review flag now uses model disagreement or MC Dropout std above 0.02, which caught 13 of 15 errors on a validation subset (13.7% of images flagged) and 11 of 14 on a test subset (13.0% flagged). Some errors are still missed.</li>',
   "limitation calibration")
rep("</ol>",
    '<li><b style="color:var(--text)">Source confounding.</b> AMD (RFMiD, ADAM) and DR (APTOS) each come only from sources that appear in no other class, so their near-perfect scores may partly reflect source or camera cues; 27 of the 28 ResNet50 errors fall in the three source-matched classes (Normal, Cataract, Glaucoma). Patient-level separation between splits is not confirmed.</li></ol>',
    "limitation sources")

# ---------------------------------------------------------------- results logic: use the new review flag
rep("const badge = document.getElementById('severityBadge');",
    "const badge = document.getElementById('severityBadge');\n    badge.removeAttribute('style');", "badge reset")
rx(r"if\(data\.is_uncertain\)\{", "if(data.needs_review){", "badge condition")
rep("🟡 Low Confidence — Review Recommended", "🟡 Flagged for Review — low confidence or model disagreement", "badge text")
rx(r"if\(d\.predicted_class\.toLowerCase\(\) === 'normal' && !d\.is_uncertain\)\{",
   "if(d.predicted_class.toLowerCase() === 'normal' && !d.needs_review){", "report risk normal")
rx(r"\} else if\(d\.is_uncertain \|\| !d\.models_agree\)\{", "} else if(d.needs_review || !d.models_agree){", "report risk uncertain")
rep("const isNormal = d.predicted_class.toLowerCase() === 'normal' && !d.is_uncertain;",
    "const isNormal = d.predicted_class.toLowerCase() === 'normal' && !d.needs_review;", "history normal")
rep("if(d.is_uncertain) flags.push('⚠️ low confidence');",
    "if(d.is_uncertain || d.confidence_pct < 70) flags.push('⚠️ low confidence');", "history flag")

# ---------------------------------------------------------------- recommendation text, report names
rep("document.getElementById('repRecommendation').textContent = 'Clinical review by a qualified eye-care professional is advised.';",
    "document.getElementById('repRecommendation').textContent = buildRecommendation(d);", "report recommendation")
rep("Recommendation: Clinical review by a qualified eye-care professional is advised.",
    "Recommendation: ${buildRecommendation(d)}", "share recommendation")
rep("  function shareReport(){",
    """  function buildRecommendation(d){
    let t;
    if(d.predicted_class.toLowerCase() === 'normal'){
      t = 'The AI did not flag an anomaly in this image. This does not rule out eye disease; please consult an eye-care professional if you have symptoms or are due for a routine eye examination.';
    } else {
      t = 'The AI found features consistent with ' + d.predicted_class + '. Please have this confirmed by a qualified eye-care professional; this result is not a diagnosis.';
    }
    if(!d.models_agree) t = 'The two AI models disagree: the primary model suggested ' + d.predicted_class + ' and the second model suggested ' + d.second_opinion_predicted_class + '. This image should be reviewed by a qualified eye-care professional before any conclusion is drawn.';
    if(d.needs_review) t += ' This result was flagged for review because of model disagreement or high uncertainty.';
    return t + ' Results are only valid for retinal or eye images of the type used in training.';
  }

  function shareReport(){""", "buildRecommendation function")

# ---------------------------------------------------------------- better error messages
rx(r"if\(!r\.ok\) throw new Error\('Server error: '\+r\.status\);\s*return r\.json\(\);",
   "if(!r.ok){ return r.json().catch(()=>({})).then(j=>{ throw new Error(j.error || ('Server error: '+r.status)); }); }\n        return r.json();",
   "predict error handling")
rx(r"We couldn't reach the analysis backend\.", "The analysis could not be completed.", "error title text")


# ---------------------------------------------------------------- live dashboard (real data from /api/stats)
for old, new, lab in [
 ('<div class="stat-num">1,284</div>', '<div class="stat-num" id="st_total">0</div>', "st total"),
 ('<div class="stat-num" style="color:var(--green)">946</div>', '<div class="stat-num" id="st_normal" style="color:var(--green)">0</div>', "st normal"),
 ('<div class="stat-num" style="color:var(--amber)">302</div>', '<div class="stat-num" id="st_abn" style="color:var(--amber)">0</div>', "st abnormal"),
 ('<div class="card-eyebrow">High Risk</div><div class="stat-num" style="color:var(--red)">36</div>', '<div class="card-eyebrow">Flagged for Review</div><div class="stat-num" id="st_flag" style="color:var(--red)">0</div>', "st flagged"),
 ('<div class="stat-num" style="color:var(--cyan)">94.2%</div>', '<div class="stat-num" id="an_conf" style="color:var(--cyan)">0%</div>', "an conf"),
 ('<div class="stat-num">2.4s</div>', '<div class="stat-num" id="an_ms">0 s</div>', "an time"),
 ('<div class="card-eyebrow">Scans This Week</div><div class="stat-num">187</div>', '<div class="card-eyebrow">Scans This Week</div><div class="stat-num" id="an_week">0</div>', "an week"),
 ('Overview<span class="placeholder-tag">EXAMPLE DATA</span>', 'Overview<span class="placeholder-tag" style="color:var(--green); border-color:var(--green);">LIVE - SCANS ON THIS MACHINE</span>', "overview tag"),
 ('Analytics<span class="placeholder-tag">EXAMPLE DATA</span>', 'Analytics<span class="placeholder-tag" style="color:var(--green); border-color:var(--green);">LIVE - SCANS ON THIS MACHINE</span>', "analytics tag"),
 ('Recent AI Analyses<span class="placeholder-tag">EXAMPLE DATA</span>', 'Recent AI Analyses<span class="placeholder-tag" style="color:var(--green); border-color:var(--green);">LIVE</span>', "recent tag"),
 ('<div class="card-eyebrow">Avg. Confidence</div>', '<div class="card-eyebrow">Avg. Confidence (primary)</div>', "conf label"),
 ('<div class="card-eyebrow">Avg. Processing Time</div>', '<div class="card-eyebrow">Avg. Analysis Time</div>', "time label"),
 ('<div class="card-eyebrow">Anomaly Distribution</div>', '<div class="card-eyebrow">Predicted Class Distribution</div>', "dist label"),
]:
    rep(old, new, lab)
rx(r'<div class="section-heading">Image Analysis Preview.*?(?=<div class="section-heading">Recent AI Analyses)', "", "remove fake preview")
rx(r'(Recent AI Analyses.*?<div class="card" style="overflow-x:auto;">)\s*<table>.*?</table>',
   '<div class="card" style="overflow-x:auto;"><table><tr><th>Scan ID</th><th>Time</th><th>Result</th><th>Confidence</th><th>Second Opinion</th><th>Review</th></tr><tbody id="recentBody"><tr><td colspan="6">No scans yet - analyze an image.</td></tr></tbody></table>', "recent table")
rep("  function populateReportsPage(){", """  function loadStats(){
    fetch('/api/stats').then(r=>r.json()).then(s=>{
      const set=(id,v)=>{const e=document.getElementById(id); if(e) e.textContent=v;};
      const normal=s.by_class['Normal']||0;
      set('st_total',s.total); set('st_normal',normal); set('st_abn',s.total-normal); set('st_flag',s.flagged);
      set('an_conf',s.avg_confidence+'%'); set('an_ms',(s.avg_ms/1000).toFixed(1)+' s');
      set('an_week',s.week.reduce((a,d)=>a+d.n,0));
      const bars=document.querySelectorAll('#page-analytics .bars');
      const draw=(el,items)=>{const m=Math.max(1,...items.map(i=>i.n));
        el.innerHTML=items.map(i=>`<div class="bar" style="height:${Math.max(4,100*i.n/m)}%;"><span>${i.day} ${i.n}</span></div>`).join('');};
      if(bars[0]) draw(bars[0], s.week);
      if(bars[1]) draw(bars[1], ['Normal','Cataract','Glaucoma','AMD','Diabetic Retinopathy'].map(k=>({day:k.slice(0,4),n:s.by_class[k]||0})));
    }).catch(()=>{});
    fetch('/api/history').then(r=>r.json()).then(rows=>{
      const body=document.getElementById('recentBody'); if(!body||!rows.length) return;
      body.innerHTML=rows.map(r=>`<tr><td class="mono">${r.scan_id}</td><td>${r.ts.slice(5,16)}</td><td>${r.predicted}</td><td class="mono">${r.confidence}%</td><td>${r.second}${r.agree?'':' (disagrees)'}</td><td><span class="badge ${r.needs_review?'badge-amber':'badge-green'}">${r.needs_review?'Flagged':'OK'}</span></td></tr>`).join('');
    }).catch(()=>{});
  }

  function populateReportsPage(){""", "loadStats function")
rep("if(key === 'reports') populateReportsPage();", "if(key === 'reports') populateReportsPage();\n    if(key === 'dashboard' || key === 'analytics') loadStats();", "goTo stats")
rep("data.scan_id = 'SCN-' + Date.now().toString().slice(-8);", "data.scan_id = 'SCN-' + Date.now().toString().slice(-8);\n    if(data.image_check && !data.image_check.looks_like_fundus) showToast(data.image_check.message, 'error', 9000);\n    setTimeout(loadStats, 600);", "image check toast")
rx(r"</script>\s*</body>", "loadStats();\n</script>\n</body>", "initial stats load")


# ---------------------------------------------------------------- working image viewer controls
rep('<div class="vctrl">➕</div><div class="vctrl">➖</div><div class="vctrl">⟳</div><div class="vctrl">⛶</div>',
    '<div class="vctrl" title="Zoom in" onclick="viewerZoom(1.25)">➕</div><div class="vctrl" title="Zoom out" onclick="viewerZoom(0.8)">➖</div>'
    '<div class="vctrl" title="Reset view" onclick="viewerReset()">⟳</div><div class="vctrl" title="Fullscreen" onclick="viewerFullscreen()">⛶</div>', "viewer buttons")
rep("</style>", ".viewer-frame img{user-select:none; -webkit-user-drag:none;} .viewer-frame.zoomed{cursor:grab;} .viewer-frame:fullscreen{background:#000;}\n</style>", "viewer css")
rep("  function showHelp(){", """  // ---------- Image viewer: zoom / pan / reset / fullscreen (original + heatmap move together) ----------
  let vz = {s:1, x:0, y:0};
  function applyView(){
    ['viewerImgOriginal','viewerImgHeatmap'].forEach(id=>{
      const e = document.getElementById(id);
      if(e) e.style.transform = `translate(${vz.x}px, ${vz.y}px) scale(${vz.s})`;
    });
    document.querySelectorAll('#resultsBlock .viewer-frame').forEach(f=>f.classList.toggle('zoomed', vz.s > 1));
  }
  function viewerZoom(f){
    vz.s = Math.min(6, Math.max(1, vz.s * f));
    if(vz.s === 1){ vz.x = 0; vz.y = 0; }
    applyView();
  }
  function viewerReset(){ vz = {s:1, x:0, y:0}; applyView(); }
  function viewerFullscreen(){
    const frame = document.getElementById('viewerImgHeatmap').parentElement;
    if(document.fullscreenElement) document.exitFullscreen();
    else if(frame.requestFullscreen) frame.requestFullscreen();
  }
  (function(){
    let drag = null;
    document.querySelectorAll('#resultsBlock .viewer-frame').forEach(frame=>{
      frame.addEventListener('mousedown', e=>{
        if(vz.s <= 1 || e.target.closest('.viewer-controls')) return;
        drag = {x:e.clientX - vz.x, y:e.clientY - vz.y}; e.preventDefault();
      });
      frame.addEventListener('wheel', e=>{ e.preventDefault(); viewerZoom(e.deltaY < 0 ? 1.15 : 0.87); }, {passive:false});
    });
    window.addEventListener('mousemove', e=>{ if(drag){ vz.x = e.clientX - drag.x; vz.y = e.clientY - drag.y; applyView(); } });
    window.addEventListener('mouseup', ()=>{ drag = null; });
  })();

  function showHelp(){""", "viewer script")
rep("document.getElementById('viewerImgHeatmap').src = data.heatmap_data_url;", "document.getElementById('viewerImgHeatmap').src = data.heatmap_data_url;\n    viewerReset();", "reset on new result")


# ---------------------------------------------------------------- batch analysis page
rep('<div class="nav-item" data-page="analyze">🔬 Analyze Image</div>',
    '<div class="nav-item" data-page="analyze">🔬 Analyze Image</div>\n      <div class="nav-item" data-page="batch">🗂️ Batch Analysis</div>', "batch nav")
rep("analyze:['Analyze Image','Dashboard / AI Analysis'],",
    "analyze:['Analyze Image','Dashboard / AI Analysis'],\n    batch:['Batch Analysis','Dashboard / Batch Analysis'],", "batch title")
rep('<section class="page" id="page-history">', """<section class="page" id="page-batch">
      <div class="section-heading">Batch Analysis</div>
      <div class="card">
        <p style="color:var(--text-dim); font-size:13px; margin-bottom:14px; line-height:1.6;">Select up to 30 fundus images. Each image is checked by both models (heatmaps are skipped in batch mode) and the results can be exported as CSV. Expect a few seconds per image on CPU. Screening aid, not a diagnosis.</p>
        <input type="file" id="batchInput" accept="image/*" multiple style="display:none">
        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
          <button class="btn btn-primary" onclick="document.getElementById('batchInput').click()">Choose Images</button>
          <button class="btn btn-ghost" id="batchRunBtn" onclick="runBatch()" disabled>Analyze Batch</button>
          <button class="btn btn-ghost" id="batchCsvBtn" onclick="downloadBatchCsv()" style="display:none;">Download CSV</button>
          <span id="batchInfo" style="font-size:12.5px; color:var(--text-dim);">No images selected</span>
        </div>
      </div>
      <div class="card" id="batchResults" style="display:none; margin-top:18px; overflow-x:auto;">
        <div id="batchSummary" style="font-size:13px; color:var(--text-dim); margin-bottom:12px;"></div>
        <table><tr><th>File</th><th>Result</th><th>Confidence</th><th>Second opinion</th><th>Review</th><th>Note</th></tr><tbody id="batchBody"></tbody></table>
      </div>
    </section>

    <section class="page" id="page-history">""", "batch page")
rep("</style>", ".btn:disabled{opacity:.5; cursor:not-allowed;}\n</style>", "disabled css")
rep("  function showHelp(){", """  // ---------- Batch analysis ----------
  let batchFiles = [], batchRows = [];
  const escHtml = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  document.getElementById('batchInput').addEventListener('change', e=>{
    batchFiles = [...e.target.files];
    document.getElementById('batchInfo').textContent = batchFiles.length ? batchFiles.length + ' image(s) selected' : 'No images selected';
    document.getElementById('batchRunBtn').disabled = batchFiles.length === 0;
  });
  function runBatch(){
    if(!batchFiles.length) return;
    if(batchFiles.length > 30){ showToast('Please select at most 30 images.', 'error'); return; }
    const btn = document.getElementById('batchRunBtn');
    btn.disabled = true; btn.textContent = 'Analyzing ' + batchFiles.length + ' images...';
    const fd = new FormData();
    batchFiles.forEach(f => fd.append('images', f));
    fetch('/predict_batch', {method:'POST', body:fd})
      .then(r => r.ok ? r.json() : r.json().catch(()=>({})).then(j => { throw new Error(j.error || ('Server error: ' + r.status)); }))
      .then(d => { batchRows = d.results; renderBatch(); showToast('Batch analysis complete.', 'success'); loadStats(); })
      .catch(err => showToast('Batch failed: ' + err.message, 'error'))
      .finally(() => { btn.disabled = false; btn.textContent = 'Analyze Batch'; });
  }
  function renderBatch(){
    const ok = batchRows.filter(r => !r.error);
    const flagged = ok.filter(r => r.needs_review).length;
    const counts = {};
    ok.forEach(r => counts[r.predicted_class] = (counts[r.predicted_class] || 0) + 1);
    document.getElementById('batchSummary').innerHTML = '<b style="color:var(--text)">' + ok.length + ' analysed</b>, ' +
      flagged + ' flagged for review, ' + (batchRows.length - ok.length) + ' failed. ' +
      Object.entries(counts).map(([k,v]) => k + ': ' + v).join(' | ');
    document.getElementById('batchBody').innerHTML = batchRows.map(r => r.error
      ? `<tr><td>${escHtml(r.filename)}</td><td colspan="4">-</td><td><span class="badge badge-red">${escHtml(r.error)}</span></td></tr>`
      : `<tr><td>${escHtml(r.filename)}</td><td>${r.predicted_class}</td><td class="mono">${r.confidence_pct}%</td><td>${r.second_opinion_predicted_class}${r.models_agree ? '' : ' (disagrees)'}</td><td><span class="badge ${r.needs_review ? 'badge-amber' : 'badge-green'}">${r.needs_review ? 'Flagged' : 'OK'}</span></td><td>${r.looks_like_fundus ? '' : 'May not be a fundus image'}</td></tr>`).join('');
    document.getElementById('batchResults').style.display = 'block';
    document.getElementById('batchCsvBtn').style.display = 'inline-flex';
  }
  function downloadBatchCsv(){
    const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
    const head = ['filename','predicted_class','confidence_pct','second_opinion','second_opinion_confidence_pct','models_agree','flagged_for_review','looks_like_fundus','error'];
    const lines = [head.join(',')].concat(batchRows.map(r => [r.filename, r.predicted_class, r.confidence_pct, r.second_opinion_predicted_class,
      r.second_opinion_confidence_pct, r.models_agree, r.needs_review, r.looks_like_fundus, r.error].map(q).join(',')));
    const url = URL.createObjectURL(new Blob([lines.join('\\n')], {type:'text/csv'}));
    const a = document.createElement('a'); a.href = url; a.download = 'eye-anomaly-batch-results.csv';
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }

  function showHelp(){""", "batch script")


# ---------------------------------------------------------------- sidebar subtitle
rep('<div class="logo-sub">Intelligent Eye Anomaly Detection</div>', '<div class="logo-sub">AI Retinal Screening</div>', "sidebar subtitle")


# ---------------------------------------------------------------- result page wording and "analyze another" button
rep('<div class="section-heading">Why did the AI flag this image?</div>', '<div class="section-heading">How did the AI reach this result?</div>', "heatmap heading")
rep('<button class="btn btn-primary" onclick="goTo(\'reports\')">Generate Clinical Report</button>',
    '<button class="btn btn-primary" onclick="goTo(\'reports\')">Generate Clinical Report</button> '
    '<button class="btn btn-ghost" onclick="resetUpload(); window.scrollTo({top:0, behavior:\'smooth\'})">Analyze Another Image</button>', "analyze another button")


# ---------------------------------------------------------------- accurate wording for why ResNet50 is primary
rep("ResNet50 is the primary model by design: MC Dropout uncertainty and Grad-CAM are built around it.",
    "ResNet50 was chosen as the primary model in the application design (the app applies MC Dropout and Grad-CAM to the primary model).", "primary model wording")

with open(FILE, "w", encoding="utf-8") as fh:
    fh.write(text)

print(f"\nApplied {applied} changes.")
if missed:
    print("NOT found (already changed, or text differs) - send these to me:")
    for m in missed:
        print("  -", m)
else:
    print("All changes applied.")
print("\nNext: python backend_app.py   then open http://127.0.0.1:5000 and press Ctrl+F5.")