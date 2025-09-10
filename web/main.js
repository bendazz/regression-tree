import { csv, json, hierarchy, tree as d3tree, select, zoom, interpolate } from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const svg = select("#svg");
const rootG = svg.append("g").attr("class", "rootG");
const gLinks = rootG.append("g").attr("class", "links");
const gNodes = rootG.append("g").attr("class", "nodes");

const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
const stepBtn = document.getElementById("step");
const speedEl = document.getElementById("speed");
const fitBtn = document.getElementById("fit");
const resetBtn = document.getElementById("reset");
const sampleIdxEl = document.getElementById("sampleIdx");
const sampleTotalEl = document.getElementById("sampleTotal");
const yTrueEl = document.getElementById("yTrue");
const yPredEl = document.getElementById("yPred");
const yErrEl = document.getElementById("yErr");
const tooltip = document.getElementById('tooltip');

let meta, model, testData, featureNames, root, layout, nodesById;
let animTimer = null;
let currentIndex = 0;
let ballsLayer, stacksLayer;

function toHier(model) {
  const nodeMap = new Map(model.nodes.map(n => [n.id, n]));
  function build(id) {
    const n = nodeMap.get(id);
    const children = n.isLeaf ? [] : [build(n.left), build(n.right)];
    return { ...n, children };
  }
  return build(model.root);
}

function computePrediction(sample) {
  let n = nodesById.get(model.root);
  const path = [n];
  while (!n.data.isLeaf) {
    const idx = n.data.featureIndex;
    const fname = featureNames[idx];
    const v = +sample[fname];
    const nextId = (v <= n.data.threshold) ? n.data.left : n.data.right;
    n = nodesById.get(nextId);
    path.push(n);
  }
  return { leaf: n, path };
}

function setActivePath(path) {
  svg.selectAll('.node').classed('active', false);
  svg.selectAll('.link').classed('active', false);
  for (let i = 0; i < path.length; i++) {
    const n = path[i];
    svg.select(`#node-${n.data.id}`).classed('active', true);
    if (i > 0) svg.select(`#link-${path[i-1].data.id}-${n.data.id}`).classed('active', true);
  }
}

function render() {
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;

  layout = d3tree().nodeSize([36, 140]); // fixed spacing for readability
  root = hierarchy(toHier(model));
  root = layout(root);

  nodesById = new Map();
  root.each(n => nodesById.set(n.data.id, n));

  // Links
  const links = root.links();
  const linkSel = gLinks.selectAll("line").data(links, d => `${d.source.data.id}-${d.target.data.id}`);
  linkSel.join(
    enter => enter.append("line")
      .attr("class", "link")
      .attr("id", d => `link-${d.source.data.id}-${d.target.data.id}`)
      .attr("x1", d => d.source.y)
      .attr("y1", d => d.source.x)
      .attr("x2", d => d.target.y)
      .attr("y2", d => d.target.x),
    update => update
      .attr("x1", d => d.source.y)
      .attr("y1", d => d.source.x)
      .attr("x2", d => d.target.y)
      .attr("y2", d => d.target.x),
    exit => exit.remove()
  );

  // Nodes
  const nodeSel = gNodes.selectAll("g.node").data(root.descendants(), d => d.data.id);
  const nodeEnter = nodeSel.join(
    enter => {
      const g = enter.append("g").attr("class", d => `node ${d.data.isLeaf ? 'leaf' : ''}`).attr("id", d => `node-${d.data.id}`)
        .attr("transform", d => `translate(${d.y}, ${d.x})`);
      g.append("circle").attr("r", 16);
      // Tooltip for both internal and leaf nodes
      g.on('mouseenter', (e, d) => {
        tooltip.style.opacity = 1;
        if (d.data.isLeaf) {
          tooltip.textContent = `pred: ${d.data.value.toFixed(3)}`;
        } else {
          const fname = featureNames[d.data.featureIndex];
          tooltip.textContent = `${fname} ≤ ${d.data.threshold.toFixed(2)}`;
        }
      }).on('mousemove', (e) => {
        const pad = 10;
        tooltip.style.left = `${e.clientX + pad}px`;
        tooltip.style.top = `${e.clientY + pad}px`;
      }).on('mouseleave', () => {
        tooltip.style.opacity = 0;
      });
      return g;
    },
    update => update.attr("transform", d => `translate(${d.y}, ${d.x})`),
    exit => exit.remove()
  );

  // Ensure layers for animation and stacks (topmost so balls show above nodes)
  if (!ballsLayer) ballsLayer = rootG.append('g').attr('class', 'balls');
  if (!stacksLayer) stacksLayer = rootG.append('g').attr('class', 'stacks');

  fitToView();
}

async function init() {
  [meta, model, testData] = await Promise.all([
    json('/model/meta.json'),
    json('/model/tree.json'),
    csv('/data/california_housing_test.csv')
  ]);
  featureNames = meta.featureNames;
  sampleTotalEl.textContent = testData.length;
  // enable zoom/pan
  svg.call(zoom().scaleExtent([0.2, 3]).on('zoom', (e) => {
    rootG.attr('transform', e.transform);
  }));
  render();
  // enable controls now that everything is ready
  playBtn.disabled = false;
  pauseBtn.disabled = false;
  stepBtn.disabled = false;
}

function updateStats(sample, leaf) {
  const y = +sample[meta.target];
  const pred = leaf.data.value;
  yTrueEl.textContent = y.toFixed(3);
  yPredEl.textContent = pred.toFixed(3);
  yErrEl.textContent = (pred - y).toFixed(3);
}

function stepOnce() {
  if (!testData || testData.length === 0 || !nodesById) return;
  if (currentIndex >= testData.length) { pause(); return; }
  const sample = testData[currentIndex];
  const res = computePrediction(sample);
  setActivePath(res.path);
  updateStats(sample, res.leaf);
  sampleIdxEl.textContent = (currentIndex + 1);
  if (ballsLayer && stacksLayer) {
  animateBallAlongPath(res.path, sample, () => stackAtLeaf(res.leaf, sample));
  }
  currentIndex++;
}

function play() {
  if (animTimer) return;
  if (!testData || currentIndex >= testData.length) return; // already finished
  const speed = +speedEl.value; // 0..100
  const interval = 100 + (1000 - 100) * (1 - speed / 100); // 100..1000ms
  animTimer = setInterval(stepOnce, interval);
}

function pause() {
  if (animTimer) {
    clearInterval(animTimer);
    animTimer = null;
  }
}

playBtn.addEventListener('click', () => { pause(); play(); });
pauseBtn.addEventListener('click', pause);
stepBtn.addEventListener('click', () => { pause(); stepOnce(); });
speedEl.addEventListener('input', () => { if (animTimer) { pause(); play(); } });

function fitToView() {
  if (!root) return;
  const nodes = root.descendants();
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minX = Math.min(...xs) - 60, maxX = Math.max(...xs) + 60;
  const minY = Math.min(...ys) - 60, maxY = Math.max(...ys) + 60;
  const vbWidth = maxY - minY;
  const vbHeight = maxX - minX;
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  const k = Math.min(width / vbWidth, height / vbHeight, 1.5);
  const tx = (width - k * (minY + maxY)) / 2;
  const ty = (height - k * (minX + maxX)) / 2;
  rootG.attr('transform', `translate(${tx},${ty}) scale(${k})`);
}

fitBtn.addEventListener('click', fitToView);

init();

// Animation helpers
function animateBallAlongPath(path, sample, onDone) {
  // Build segments from source->target positions
  const segments = [];
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i+1];
    segments.push({
      ax: a.y, ay: a.x,
      bx: b.y, by: b.x,
      id: `${a.data.id}-${b.data.id}`
    });
  }
  if (segments.length === 0) { onDone && onDone(); return; }

  const ball = ballsLayer.append('circle').attr('r', 8).attr('fill', '#1f77b4').attr('stroke', '#fff').attr('stroke-width', 1)
    .on('mouseenter', (e) => {
      tooltip.style.opacity = 1;
      tooltip.textContent = `y: ${(+sample[meta.target]).toFixed(3)}`;
    })
    .on('mousemove', (e) => {
      const pad = 10; tooltip.style.left = `${e.clientX + pad}px`; tooltip.style.top = `${e.clientY + pad}px`;
    })
    .on('mouseleave', () => { tooltip.style.opacity = 0; });

  let segIndex = 0;
  const speedPxPerSec = 280; // visual speed along edges

  function moveNext() {
    if (segIndex >= segments.length) { ball.remove(); onDone && onDone(); return; }
    const { ax, ay, bx, by } = segments[segIndex];
    const dist = Math.hypot(bx - ax, by - ay);
    const duration = (dist / speedPxPerSec) * 1000;
    const ix = interpolate(ax, bx);
    const iy = interpolate(ay, by);
    const start = performance.now();

    function frame(t) {
      const k = Math.min(1, (t - start) / duration);
      const x = ix(k), y = iy(k);
      ball.attr('transform', `translate(${x},${y})`);
      if (k < 1) requestAnimationFrame(frame); else { segIndex++; moveNext(); }
    }
    requestAnimationFrame(frame);
  }
  moveNext();
}

function stackAtLeaf(leafNode, sample) {
  // Stack balls in a small grid near the leaf node position
  const key = `stack-${leafNode.data.id}`;
  let g = stacksLayer.select(`#${key}`);
  if (g.empty()) {
    g = stacksLayer.append('g').attr('id', key).attr('transform', `translate(${leafNode.y + 20}, ${leafNode.x - 20})`);
  }
  const existing = g.selectAll('circle').nodes().length;
  const cols = 6;
  const size = 11; // spacing for larger dots
  const cx = (existing % cols) * size;
  const cy = Math.floor(existing / cols) * size;
  g.append('circle')
    .attr('cx', cx).attr('cy', cy).attr('r', 5)
    .attr('fill', '#1f77b4').attr('stroke', '#fff').attr('stroke-width', 0.5)
    .on('mouseenter', (e) => {
      tooltip.style.opacity = 1; tooltip.textContent = `y: ${(+sample[meta.target]).toFixed(3)}`;
    })
    .on('mousemove', (e) => {
      const pad = 10; tooltip.style.left = `${e.clientX + pad}px`; tooltip.style.top = `${e.clientY + pad}px`;
    })
    .on('mouseleave', () => { tooltip.style.opacity = 0; });
}

resetBtn.addEventListener('click', () => {
  if (ballsLayer) ballsLayer.selectAll('*').remove();
  if (stacksLayer) stacksLayer.selectAll('*').remove();
  currentIndex = 0;
  sampleIdxEl.textContent = 0;
  // allow playing again after reset
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
});
