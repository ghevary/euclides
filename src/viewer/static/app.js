/* Possession Viewer. Plain JavaScript, no framework, no outside request.
   Read-only by design: nothing on this page edits a label or computes a score. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var LABELS = ["H", "A", "D", "X"];
  var CODE = { H: 0, A: 1, D: 2, X: 3 };
  var NAMES = { H: "H · Home in possession", A: "A · Away in possession", D: "D · Dead ball", X: "X · Not graded" };
  var CLASS = { H: "lh", A: "la", D: "ld", X: "lx" };
  var COLORS = ["#e5484d", "#3b82f6", "#8b95a3", "#343b46"];
  var TRACK_BLOCK = 128;   // frames per tracking request
  var TRACK_KEEP = 48;     // blocks kept in memory
  var ZOOM_HALF = 250;     // frames each side of the cursor in the detail strip

  var video = $("video");
  var S = {
    meta: null, fps: 25, n: 0,       // n: frames of the tactical timeline (rows of a submission)
    total: 0,                        // frames of the video being played
    frame: -1, target: null,
    source: "tactical", synced: true,
    A: null, B: null, diff: null,    // label sets: {name, codes, segs}
    tags: [],
    overlay: true,
    blocks: new Map(), pending: new Set(), trackState: "none",
    objectUrl: null, lastCallback: 0
  };

  // ------------------------------------------------------------------ labels

  function setFromSegments(name, segs, n) {
    var codes = new Uint8Array(n).fill(CODE.X);
    segs.forEach(function (s) { codes.fill(CODE[s[2]], s[0], s[1] + 1); });
    return { name: name, codes: codes, segs: segs };
  }

  function segmentsOf(codes) {
    var segs = [], start = 0;
    for (var i = 1; i <= codes.length; i++) {
      if (i === codes.length || codes[i] !== codes[start]) {
        segs.push([start, i - 1, LABELS[codes[start]]]);
        start = i;
      }
    }
    return segs;
  }

  /* Same contract as SUBMISSION.md: header frame,label; frame = row number; label in H A D X. */
  function parseCsv(text, n) {
    var lines = text.split(/\r?\n/);
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    var head = (lines[0] || "").split(",").map(function (c) { return c.trim().toLowerCase(); });
    if (head.length !== 2 || head[0] !== "frame" || head[1] !== "label") throw new Error("header must be 'frame,label'");
    if (lines.length - 1 !== n) throw new Error((lines.length - 1) + " data rows; expected " + n);
    var codes = new Uint8Array(n);
    for (var i = 0; i < n; i++) {
      var cells = lines[i + 1].split(",");
      if (cells.length !== 2 || cells[0].trim() !== String(i)) throw new Error("row " + i + ": frame column must be " + i);
      var code = CODE[cells[1].trim().toUpperCase()];
      if (code === undefined) throw new Error("row " + i + ": label must be H, A, D or X");
      codes[i] = code;
    }
    return codes;
  }

  function segIndex(segs, frame) {
    var lo = 0, hi = segs.length - 1;
    while (lo < hi) {
      var mid = (lo + hi + 1) >> 1;
      if (segs[mid][0] <= frame) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  function computeDiff() {
    S.diff = null;
    if (!S.A || !S.B) return;
    var runs = [], a = S.A.codes, b = S.B.codes, start = -1;
    for (var i = 0; i <= S.n; i++) {
      var differs = i < S.n && a[i] !== b[i];
      if (differs && start < 0) start = i;
      if (!differs && start >= 0) { runs.push([start, i - 1]); start = -1; }
    }
    S.diff = runs;
  }

  function summarize(set) {
    var counts = [0, 0, 0, 0], spells = 0;
    set.segs.forEach(function (s) {
      counts[CODE[s[2]]] += s[1] - s[0] + 1;
      if (s[2] === "H" || s[2] === "A") spells++;
    });
    return { counts: counts, spells: spells, mean: spells ? (counts[0] + counts[1]) / S.fps / spells : 0 };
  }

  // ------------------------------------------------------------------ time and frames

  function clampFrame(f) { return Math.max(0, Math.min(S.total - 1, f)); }

  /* Seek to the middle of frame f, so rounding in the player cannot land on f-1 or f+1. */
  function seekTo(f) {
    if (!S.total) return;
    f = clampFrame(Math.round(f));
    S.target = f;
    video.currentTime = (f + 0.5) / S.fps;
  }

  function step(delta) {
    if (!video.paused) video.pause();
    var base = S.target !== null ? S.target : Math.max(0, S.frame);
    seekTo(base + delta);
  }

  function onFrame(f) {
    f = clampFrame(f);
    if (f === S.frame) return;
    S.frame = f;
    render();
  }

  function startFrameClock() {
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      // mediaTime is the timestamp of the frame actually on screen.
      var tick = function (now, info) {
        S.lastCallback = performance.now();
        if (!video.seeking) onFrame(Math.round(info.mediaTime * S.fps));
        video.requestVideoFrameCallback(tick);
      };
      video.requestVideoFrameCallback(tick);
    } else {
      var loop = function () {
        onFrame(Math.floor(video.currentTime * S.fps + 1e-3));
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }

  function fmtTime(frame) {
    var sec = Math.floor(frame / S.fps), ff = frame % S.fps;
    var mm = Math.floor(sec / 60), ss = sec % 60;
    return String(mm).padStart(2, "0") + ":" + String(ss).padStart(2, "0") + "." + String(ff).padStart(2, "0");
  }

  function halfName(frame) {
    var seg = S.meta.segments;
    for (var i = 0; i < seg.length; i++) {
      if (frame < seg[i][0]) return i === 0 ? "Before the match" : "Halftime";
      if (frame <= seg[i][1]) return "Half " + (i + 1);
    }
    return "After the match";
  }

  // ------------------------------------------------------------------ panel

  function paintLabel(el, letter) {
    el.className = el.className.replace(/\bl[hadx]\b/g, "").trim() + " " + (letter ? CLASS[letter] : "lx");
    el.textContent = letter ? NAMES[letter] : "–";
  }

  function render() {
    var f = S.frame;
    $("frameNo").textContent = f;
    $("hudFrame").textContent = f;
    $("time").textContent = fmtTime(f);
    $("half").textContent = S.synced ? halfName(f) : "–";

    var la = S.synced && S.A ? LABELS[S.A.codes[f]] : null;
    paintLabel($("labelA"), la);
    var hud = $("hudLabel");
    hud.textContent = la || "–";
    hud.style.background = la ? COLORS[CODE[la]] : "transparent";
    if (la) {
      var s = S.A.segs[segIndex(S.A.segs, f)];
      $("segA").textContent = "segment " + s[0] + "–" + s[1] + " · " + ((s[1] - s[0] + 1) / S.fps).toFixed(2) + " s";
    } else {
      $("segA").textContent = S.synced && !S.A ? "No label file yet. Use “Open CSV…”." : "";
    }
    if (S.B && S.synced) {
      var lb = LABELS[S.B.codes[f]];
      paintLabel($("labelB"), lb);
      $("agree").textContent = la === null ? "" : (la === lb ? "same as the main labels" : "DIFFERENT from the main labels");
    }

    var total = Math.max(1, S.total);
    $("tlCursor").style.left = ((f + 0.5) / total * 100) + "%";
    drawZoom();
    renderTracking();
  }

  function renderTracking() {
    var f = S.frame, item = S.synced ? trackAt(f) : null;
    $("ballSeen").textContent = item ? (item.b ? "yes" : "no") : "–";
    $("distH").textContent = item && item.dh !== null ? item.dh.toFixed(1) + " m" : "–";
    $("distA").textContent = item && item.da !== null ? item.da.toFixed(1) + " m" : "–";
    var tag = "–";
    if (S.synced) {
      for (var i = 0; i < S.tags.length; i++) {
        var t = S.tags[i];
        if (t.start > f) break;
        if (f <= t.end) tag = (t.type || "set piece") + (t.team ? " (" + t.team + ")" : "");
      }
    }
    $("tagNow").textContent = tag;
    drawOverlay(item);
    drawPitch(item);
    if (S.synced && S.trackState === "ready") {
      wantBlock(f);
      if (!video.paused) wantBlock(f + TRACK_BLOCK);
    }
  }

  // ------------------------------------------------------------------ tracking blocks

  function trackAt(f) {
    var block = S.blocks.get(Math.floor(f / TRACK_BLOCK));
    return block ? block[f % TRACK_BLOCK] || null : null;
  }

  function wantBlock(f) {
    if (f < 0 || f >= S.n) return;
    var key = Math.floor(f / TRACK_BLOCK);
    if (S.blocks.has(key) || S.pending.has(key)) return;
    S.pending.add(key);
    fetch("/api/track?start=" + key * TRACK_BLOCK + "&count=" + TRACK_BLOCK)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        S.pending.delete(key);
        if (data.state !== "ready") return;
        S.blocks.set(key, data.frames);
        if (S.blocks.size > TRACK_KEEP) S.blocks.delete(S.blocks.keys().next().value);
        if (Math.floor(S.frame / TRACK_BLOCK) === key) renderTracking();
      })
      .catch(function () { S.pending.delete(key); });
  }

  function pollTracking() {
    fetch("/api/meta").then(function (r) { return r.json(); }).then(function (meta) {
      S.trackState = meta.tracking;
      showTrackState(meta);
      if (meta.tracking === "loading") setTimeout(pollTracking, 2000);
      else if (meta.tracking === "ready") renderTracking();
    }).catch(function () { setTimeout(pollTracking, 4000); });
  }

  function showTrackState(meta) {
    var msg = "";
    if (meta.tracking === "loading") msg = "Loading tracking… (1–2 minutes the first time, from the cache afterwards)";
    else if (meta.tracking === "error") msg = "Tracking failed to load: " + meta.tracking_error;
    else if (meta.tracking === "none") msg = "Tracking not loaded.";
    $("trackMsg").textContent = msg;
    $("trackMsg").classList.toggle("err", meta.tracking === "error");
  }

  // ------------------------------------------------------------------ drawing

  function fitCanvas(canvas, cssHeight) {
    var ratio = window.devicePixelRatio || 1;
    var w = Math.max(1, Math.round(canvas.clientWidth * ratio));
    var h = Math.max(1, Math.round((cssHeight || canvas.clientHeight) * ratio));
    if (cssHeight) canvas.style.height = cssHeight + "px";
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return ratio;
  }

  /* Rectangle of the picture inside the <video> box (object-fit: contain). */
  function pictureRect() {
    var bw = video.clientWidth, bh = video.clientHeight;
    var vw = video.videoWidth || 16, vh = video.videoHeight || 9;
    var scale = Math.min(bw / vw, bh / vh);
    var w = vw * scale, h = vh * scale;
    return { x: (bw - w) / 2, y: (bh - h) / 2, w: w, h: h };
  }

  function drawOverlay(item) {
    var canvas = $("overlay"), ratio = fitCanvas(canvas);
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!S.overlay || !item || !item.H || !S.synced) return;
    var rect = pictureRect(), H = item.H, size = S.meta.homography_size;
    var sx = rect.w / size[0] * ratio, sy = rect.h / size[1] * ratio;
    var ox = rect.x * ratio, oy = rect.y * ratio;
    var radius = Math.max(4, rect.w / 190) * ratio;
    function project(x, y) {
      var w = H[6] * x + H[7] * y + H[8];
      if (!w) return null;
      return [ox + (H[0] * x + H[1] * y + H[2]) / w * sx, oy + (H[3] * x + H[4] * y + H[5]) / w * sy];
    }
    function ring(points, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2 * ratio;
      for (var i = 0; i < points.length; i += 2) {
        var p = project(points[i], points[i + 1]);
        if (!p) continue;
        ctx.beginPath(); ctx.ellipse(p[0], p[1], radius, radius * 0.45, 0, 0, Math.PI * 2); ctx.stroke();
      }
    }
    ring(item.h, COLORS[0]);
    ring(item.a, COLORS[1]);
    if (item.b) {
      var p = project(item.b[0], item.b[1]);
      if (p) {
        ctx.fillStyle = "#ffe14d"; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5 * ratio;
        ctx.beginPath(); ctx.arc(p[0], p[1], radius * 0.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }
  }

  function drawPitch(item) {
    var canvas = $("pitch"), ctx = canvas.getContext("2d");
    var W = canvas.width, Hh = canvas.height, m = 8;
    ctx.clearRect(0, 0, W, Hh);
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1;
    ctx.strokeRect(m, m, W - 2 * m, Hh - 2 * m);
    ctx.beginPath(); ctx.moveTo(W / 2, m); ctx.lineTo(W / 2, Hh - m); ctx.stroke();
    ctx.beginPath(); ctx.arc(W / 2, Hh / 2, (Hh - 2 * m) * 0.135, 0, Math.PI * 2); ctx.stroke();
    if (!item) return;
    function dots(points, color, r) {
      ctx.fillStyle = color;
      for (var i = 0; i < points.length; i += 2) {
        ctx.beginPath();
        ctx.arc(m + points[i] * (W - 2 * m), m + points[i + 1] * (Hh - 2 * m), r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    dots(item.h, COLORS[0], 4);
    dots(item.a, COLORS[1], 4);
    if (item.b) { dots(item.b, "#000", 4.5); dots(item.b, "#ffe14d", 3); }
  }

  var ROW = { tags: 8, set: 20, gap: 2, diff: 6 };

  function drawTimeline() {
    var canvas = $("tl");
    var twoRows = S.synced && S.B;
    var height = ROW.tags + ROW.set + (twoRows ? ROW.gap + ROW.set + ROW.gap + ROW.diff : 0);
    var ratio = fitCanvas(canvas, height), ctx = canvas.getContext("2d");
    var W = canvas.width, total = Math.max(1, S.total);
    ctx.clearRect(0, 0, W, canvas.height);
    function x(frame) { return frame / total * W; }
    function row(set, top, h) {
      set.segs.forEach(function (s) {
        var a = Math.floor(x(s[0])), b = Math.ceil(x(s[1] + 1));
        ctx.fillStyle = COLORS[CODE[s[2]]];
        ctx.fillRect(a, top * ratio, Math.max(1, b - a), h * ratio);
      });
    }
    var top = ROW.tags;
    if (!S.synced || !S.A) {
      ctx.fillStyle = COLORS[3];
      ctx.fillRect(0, top * ratio, W, ROW.set * ratio);
    } else {
      row(S.A, top, ROW.set);
    }
    if (S.synced) {
      ctx.fillStyle = "#f2c14e";
      S.tags.forEach(function (t) {
        ctx.fillRect(Math.floor(x(t.start)), 0, Math.max(2 * ratio, x(t.end) - x(t.start)), (ROW.tags - 2) * ratio);
      });
    }
    if (twoRows) {
      top += ROW.set + ROW.gap;
      row(S.B, top, ROW.set);
      top += ROW.set + ROW.gap;
      ctx.fillStyle = "#ffffff";
      (S.diff || []).forEach(function (d) {
        var a = Math.floor(x(d[0]));
        ctx.fillRect(a, top * ratio, Math.max(1, Math.ceil(x(d[1] + 1)) - a), ROW.diff * ratio);
      });
    }
  }

  function drawZoom() {
    var canvas = $("zoom");
    var twoRows = S.synced && S.B;
    var ratio = fitCanvas(canvas, twoRows ? 46 : 26), ctx = canvas.getContext("2d");
    var W = canvas.width, Hc = canvas.height, f = Math.max(0, S.frame);
    var span = 2 * ZOOM_HALF + 1, first = f - ZOOM_HALF, px = W / span;
    ctx.clearRect(0, 0, W, Hc);
    function row(set, top, h) {
      var lo = Math.max(0, first), hi = Math.min(S.n - 1, f + ZOOM_HALF);
      if (lo > hi) return;
      for (var i = segIndex(set.segs, lo); i < set.segs.length && set.segs[i][0] <= hi; i++) {
        var s = set.segs[i], a = Math.max(lo, s[0]), b = Math.min(hi, s[1]);
        ctx.fillStyle = COLORS[CODE[s[2]]];
        ctx.fillRect(Math.floor((a - first) * px), top, Math.ceil((b - a + 1) * px), h);
      }
    }
    var rowH = twoRows ? (Hc - 8 * ratio) / 2 : Hc - 6 * ratio;
    if (S.synced && S.A) row(S.A, 0, rowH);
    if (twoRows) row(S.B, rowH + 2 * ratio, rowH);
    // one tick per second, counted from frame 0
    ctx.fillStyle = "rgba(255,255,255,.5)";
    for (var t = Math.ceil(Math.max(0, first) / S.fps) * S.fps; t <= f + ZOOM_HALF; t += S.fps) {
      ctx.fillRect(Math.floor((t - first) * px), Hc - 5 * ratio, Math.max(1, ratio), 5 * ratio);
    }
    var cx = (ZOOM_HALF + 0.5) * px;
    ctx.fillStyle = "#fff"; ctx.fillRect(Math.round(cx - ratio), 0, 2 * ratio, Hc);
  }

  function redrawAll() {
    drawTimeline();
    if (S.frame >= 0) render();
  }

  // ------------------------------------------------------------------ label files

  function refreshLabelUi() {
    $("nameA").textContent = S.A ? S.A.name : "";
    $("nameB").textContent = S.B ? S.B.name : "";
    $("rowB").hidden = !(S.B && S.synced);
    $("clearB").hidden = !S.B;
    $("diffNav").hidden = !(S.B && S.A);
    $("unsynced").hidden = S.synced;
    var rows = "";
    if (S.A) {
      var sum = summarize(S.A);
      rows += LABELS.map(function (l, i) { return "<tr><td>" + l + "</td><td>" + sum.counts[i] + " frames</td></tr>"; }).join("");
      rows += "<tr><td>Possession (H/A)</td><td>" + sum.spells + "</td></tr>";
      rows += "<tr><td>Mean length</td><td>" + sum.mean.toFixed(1) + " s</td></tr>";
    }
    $("summary").innerHTML = rows;
    redrawAll();
  }

  function loadCsvFile(file, slot) {
    var msg = $("fileMsg");
    msg.classList.remove("err");
    msg.textContent = "Reading " + file.name + "…";
    file.text().then(function (text) {
      var codes = parseCsv(text, S.n);
      S[slot] = { name: file.name, codes: codes, segs: segmentsOf(codes) };
      computeDiff();
      msg.textContent = slot === "A" ? "Main labels: " + file.name : "Compared with: " + file.name +
        " · " + (S.diff ? S.diff.length : 0) + " differing ranges";
      refreshLabelUi();
    }).catch(function (error) {
      msg.classList.add("err");
      msg.textContent = file.name + " rejected: " + error.message;
    });
  }

  // ------------------------------------------------------------------ navigation

  function jumpChange(direction) {
    if (!S.A || !S.synced) return;
    var segs = S.A.segs, f = S.target !== null ? S.target : S.frame, i = segIndex(segs, f);
    if (direction > 0) { if (i + 1 < segs.length) seekTo(segs[i + 1][0]); }
    else if (f > segs[i][0]) seekTo(segs[i][0]);
    else if (i > 0) seekTo(segs[i - 1][0]);
    video.pause();
  }

  function jumpDiff(direction) {
    if (!S.diff || !S.diff.length) return;
    var f = S.target !== null ? S.target : S.frame, pick = null;
    if (direction > 0) {
      for (var i = 0; i < S.diff.length; i++) if (S.diff[i][0] > f) { pick = S.diff[i][0]; break; }
    } else {
      for (var j = S.diff.length - 1; j >= 0; j--) if (S.diff[j][0] < f) { pick = S.diff[j][0]; break; }
    }
    if (pick !== null) { video.pause(); seekTo(pick); }
  }

  function frameAtPointer(event, wrap, total, first) {
    var rect = wrap.getBoundingClientRect();
    var share = Math.max(0, Math.min(0.999999, (event.clientX - rect.left) / rect.width));
    return first + Math.floor(share * total);
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else $("stage").requestFullscreen();
  }

  function togglePlay() {
    if (video.paused) video.play().catch(function () {}); else video.pause();
  }

  // ------------------------------------------------------------------ sources

  function setSource(source, url) {
    var keep = S.source === source ? S.frame : 0;
    S.source = source;
    S.frame = -1; S.target = null;
    video.pause();
    video.src = url;
    video.load();
    video.addEventListener("loadedmetadata", function once() {
      video.removeEventListener("loadedmetadata", once);
      var frames = Math.round(video.duration * S.fps);
      // A video shares the label timeline only if it has the tactical frame count.
      S.synced = source === "tactical" || Math.abs(frames - S.n) <= 2;
      S.total = S.synced ? S.n : frames;
      video.playbackRate = parseFloat($("speed").value);
      refreshLabelUi();
      seekTo(keep);
    });
  }

  // ------------------------------------------------------------------ start

  function bind() {
    $("play").addEventListener("click", togglePlay);
    video.addEventListener("click", togglePlay);
    video.addEventListener("dblclick", toggleFullscreen);
    video.addEventListener("play", function () { S.target = null; $("play").textContent = "⏸ Pause"; });
    video.addEventListener("pause", function () {
      $("play").textContent = "▶ Play";
      // Snap to a whole frame, so the picture and the frame number cannot disagree while paused.
      if (S.target === null && S.total) seekTo(Math.floor(video.currentTime * S.fps + 1e-3));
    });
    video.addEventListener("timeupdate", function () {
      // Fallback for a hidden tab, where video frame callbacks are throttled.
      if (!video.paused && performance.now() - S.lastCallback > 500) {
        onFrame(Math.floor(video.currentTime * S.fps + 1e-3));
      }
    });
    video.addEventListener("seeked", function () {
      // Seeks go to the middle of a frame, so floor() names the frame now on screen. This also
      // covers a hidden tab, where the browser does not run video frame callbacks.
      onFrame(Math.floor(video.currentTime * S.fps + 1e-3));
      if (S.target !== null && Math.abs(video.currentTime - (S.target + 0.5) / S.fps) < 1e-3) S.target = null;
    });
    video.addEventListener("error", function () {
      var box = $("stageMsg");
      box.hidden = false;
      box.textContent = "The browser cannot play this video. Try Chrome or Edge, or make an H.264 720p proxy with the same number of frames and start with --video.";
    });
    video.addEventListener("loadeddata", function () { $("stageMsg").hidden = true; });

    $("back1").addEventListener("click", function () { step(-1); });
    $("fwd1").addEventListener("click", function () { step(1); });
    $("prevChange").addEventListener("click", function () { jumpChange(-1); });
    $("nextChange").addEventListener("click", function () { jumpChange(1); });
    $("prevDiff").addEventListener("click", function () { jumpDiff(-1); });
    $("nextDiff").addEventListener("click", function () { jumpDiff(1); });
    $("fullscreen").addEventListener("click", toggleFullscreen);
    $("speed").addEventListener("change", function () { video.playbackRate = parseFloat(this.value); });
    $("overlayToggle").addEventListener("change", function () { S.overlay = this.checked; renderTracking(); });

    $("jumpForm").addEventListener("submit", function (event) {
      event.preventDefault();
      var text = $("jump").value.trim();
      if (!/^\d+$/.test(text)) { $("jump").select(); return; }
      video.pause();
      seekTo(parseInt(text, 10));
      $("jump").blur();
    });

    $("loadA").addEventListener("click", function () { $("fileA").click(); });
    $("loadB").addEventListener("click", function () { $("fileB").click(); });
    $("fileA").addEventListener("change", function () { if (this.files[0]) loadCsvFile(this.files[0], "A"); this.value = ""; });
    $("fileB").addEventListener("change", function () { if (this.files[0]) loadCsvFile(this.files[0], "B"); this.value = ""; });
    $("clearB").addEventListener("click", function () { S.B = null; S.diff = null; $("fileMsg").textContent = ""; refreshLabelUi(); });

    $("source").addEventListener("change", function () {
      if (this.value !== "file") setSource(this.value, "/video/" + this.value);
    });
    $("openVideo").addEventListener("click", function () { $("videoFile").click(); });
    $("videoFile").addEventListener("change", function () {
      var file = this.files[0];
      this.value = "";
      if (!file) return;
      if (S.objectUrl) URL.revokeObjectURL(S.objectUrl);
      S.objectUrl = URL.createObjectURL(file);   // stays in this browser: nothing is uploaded
      var select = $("source"), option = select.querySelector('option[value="file"]');
      if (!option) { option = document.createElement("option"); option.value = "file"; select.appendChild(option); }
      option.textContent = "File: " + file.name;
      select.value = "file";
      setSource("file", S.objectUrl);
    });

    var tl = $("tlWrap"), dragging = false;
    function tlSeek(event) { seekTo(frameAtPointer(event, tl, S.total, 0)); }
    tl.addEventListener("pointerdown", function (event) { dragging = true; tl.setPointerCapture(event.pointerId); video.pause(); tlSeek(event); });
    tl.addEventListener("pointermove", function (event) {
      var f = frameAtPointer(event, tl, S.total, 0);
      var tip = "frame " + f + " · " + fmtTime(f);
      if (S.synced && S.A) tip += " · " + LABELS[S.A.codes[f]];
      if (S.synced && S.B) tip += " / " + LABELS[S.B.codes[f]];
      $("tlTip").textContent = tip;
      if (dragging) tlSeek(event);
    });
    tl.addEventListener("pointerup", function () { dragging = false; });
    tl.addEventListener("pointerleave", function () { $("tlTip").textContent = ""; });
    $("zoomWrap").addEventListener("click", function (event) {
      video.pause();
      seekTo(frameAtPointer(event, this, 2 * ZOOM_HALF + 1, S.frame - ZOOM_HALF));
    });

    document.addEventListener("keydown", function (event) {
      var tag = (event.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") {
        if (event.key === "Escape") event.target.blur();
        return;
      }
      if (event.altKey || event.metaKey) return;
      var big = event.ctrlKey ? 10 * S.fps : event.shiftKey ? S.fps : 1;
      switch (event.key) {
        case " ": togglePlay(); break;
        case "ArrowLeft": step(-big); break;
        case "ArrowRight": step(big); break;
        case "[": jumpChange(-1); break;
        case "]": jumpChange(1); break;
        case "f": case "F": toggleFullscreen(); break;
        case "g": case "G": $("jump").focus(); $("jump").select(); break;
        case "o": case "O": $("overlayToggle").click(); break;
        default: return;
      }
      event.preventDefault();
    });

    window.addEventListener("resize", redrawAll);
    document.addEventListener("fullscreenchange", function () {
      $("fullscreen").textContent = document.fullscreenElement ? "Exit fullscreen" : "⛶ Fullscreen";
      setTimeout(redrawAll, 50);
    });
  }

  function start() {
    Promise.all([
      fetch("/api/meta").then(function (r) { return r.json(); }),
      fetch("/api/labels").then(function (r) { return r.json(); }),
      fetch("/api/tags").then(function (r) { return r.json(); })
    ]).then(function (res) {
      var meta = res[0];
      S.meta = meta; S.fps = meta.fps; S.n = meta.n_frames; S.total = meta.n_frames;
      S.trackState = meta.tracking;
      if (res[1].name) S.A = setFromSegments(res[1].name, res[1].segments, S.n);
      S.tags = res[2].set_pieces || [];
      var titles = { tactical: "Tactical (labels follow it)", broadcast: "Broadcast (own timeline)" };
      meta.sources.forEach(function (name) {
        var option = document.createElement("option");
        option.value = name; option.textContent = titles[name] || name;
        $("source").appendChild(option);
      });
      bind();
      showTrackState(meta);
      if (meta.tracking === "loading") setTimeout(pollTracking, 2000);
      startFrameClock();
      setSource("tactical", "/video/tactical");
    }).catch(function (error) {
      $("stageMsg").hidden = false;
      $("stageMsg").textContent = "Could not load data from the local server: " + error.message;
    });
  }

  start();
})();
