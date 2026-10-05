/* ============================================================================
 * brain-viz.js — live 3D "fly CNS" panel for the connectome-seeded fly brain.
 *
 * WHAT YOU SEE: a schematic Drosophila central nervous system in its classic
 * dissected pose — the brain on top (two optic lobes, central brain with
 * antennal lobes and mushroom-body calyces, gnathal ganglion), the neck
 * connective, and the ventral nerve cord below (T1 / T2 / T3 leg neuromeres,
 * abdominal ganglion). This circuit is the escape / jump / leg-motor circuit,
 * which lives mostly in the nerve cord, so the whole CNS is drawn, not just
 * the head.
 *
 * WHERE EACH NEURON GOES: by cell-type identity, region-level only.
 *   sensory afferents (TA*, one group per game feature) -> optic lobes
 *   giant fibres GF / DNp01, DNp05                      -> posterior central brain
 *   DNg02 / DNg04                                      -> gnathal ganglion
 *   PSI interneurons, TTM + DLM (jump/flight) MNs      -> T2 (mesothoracic)
 *   local interneurons LN*, leg motor neurons MN*      -> T1 / T2 / T3 leg neuropils
 * Positions inside a region are a deterministic per-name scatter. HONESTY:
 * this is a schematic anatomy for legibility — the curated dataset has no
 * real soma coordinates. If a live fetch ever provides real soma positions
 * (metadata without the "ILLUSTRATIVE" note), those are used instead,
 * normalised into the brain.
 *
 * LIVE ACTIVITY (polled from flyBrain.getHidden() at ~20 Hz):
 *   - neurons glow (core + additive halo) in proportion to |activation|,
 *     normalised per pool so each layer's "most active" reads as hot
 *   - the region a hot neuron sits in lights up (fresnel shell glow)
 *   - impulses travel along the outgoing connections of firing neurons,
 *     brain -> neck -> nerve cord, so you can watch a signal cross the CNS
 *   - the giant-fibre axons (soma in the brain, terminals in T2) flare when
 *     GF fires; the top firing neurons get name labels
 *
 * PERFORMANCE: nodes are one InstancedMesh, halos/impulses are two Points
 * clouds, connections one LineSegments with per-vertex colour. No per-frame
 * allocations in the hot path.
 *
 * USAGE (all games):
 *   const viz = flyBrainViz.create({ container: document.getElementById("brain-panel") });
 *   viz.update(dt); viz.render();      // each rAF
 *   viz.setVisible(bool); viz.rebuild();
 * ========================================================================== */
(function () {
  "use strict";

  if (typeof THREE === "undefined") {
    console.warn("brain-viz: three.js not loaded — brain panel disabled.");
    return;
  }

  const ROLE_HEX = { sensor: 0xffb347, interneuron: 0xb98cff, motor: 0x4dff9a };
  const ROLE_COLORS = {
    sensor: new THREE.Color(ROLE_HEX.sensor),
    interneuron: new THREE.Color(ROLE_HEX.interneuron),
    motor: new THREE.Color(ROLE_HEX.motor),
  };

  /* visual constants */
  const NODE_SIZE = 0.2;           // base node radius (world units)
  const HOT_SCALE = 2.4;           // node scale when fully active
  const DECAY = 2.2;               // brightness fall per second
  const MAX_EDGES = 160;           // connections drawn (strongest + 1 per neuron)
  const CURVE_SEG = 10;            // segments per curved connection
  const MAX_IMPULSES = 220;
  const IMPULSE_LIFE = 0.55;       // seconds to travel one connection
  const MAX_PULSES = 16;
  const PULSE_LIFE = 0.6;
  const POLL_MS = 50;              // activation pull rate (~20 Hz)
  const ACTIVE = 0.3;              // brightness above which a node counts as lit
  const LOOK_AT = new THREE.Vector3(0, -2.2, 0);
  const CAM_DIST = 30;

  /* ---------------------------------------------------------- anatomy --- */
  /* Region shells (centre, radii) in a frontal view: +y up, +z toward the
     viewer. Fly's left is drawn on the viewer's right (+x), as in a dissected
     CNS photographed from the front. `hot` is the glow tint. */
  const REGIONS = {
    OL_L: { label: "optic lobe", hot: ROLE_HEX.sensor, parts: [
      { c: [7.3, 5.2, -0.3], r: [2.4, 3.3, 2.0] },          // medulla+lobula complex
      { c: [6.2, 5.1, -0.2], r: [1.3, 2.3, 1.4], inner: true } ] },
    OL_R: { label: "optic lobe", hot: ROLE_HEX.sensor, parts: [
      { c: [-7.3, 5.2, -0.3], r: [2.4, 3.3, 2.0] },
      { c: [-6.2, 5.1, -0.2], r: [1.3, 2.3, 1.4], inner: true } ] },
    CB: { label: "central brain", hot: ROLE_HEX.interneuron, parts: [
      { c: [0, 5.5, 0], r: [4.7, 2.9, 2.4] },
      { c: [1.35, 4.0, 1.7], r: [1.0, 0.9, 0.8], inner: true },   // antennal lobes
      { c: [-1.35, 4.0, 1.7], r: [1.0, 0.9, 0.8], inner: true },
      { c: [2.3, 7.4, -1.2], r: [1.1, 0.8, 0.9], inner: true },   // mushroom-body calyces
      { c: [-2.3, 7.4, -1.2], r: [1.1, 0.8, 0.9], inner: true } ] },
    GNG: { label: "gnathal ganglion", hot: ROLE_HEX.interneuron, parts: [
      { c: [0, 2.6, 0.5], r: [2.0, 1.2, 1.5] } ] },
    NECK: { label: "neck connective", hot: ROLE_HEX.interneuron, parts: [
      { c: [0, 0.55, 0.2], r: [0.75, 1.7, 0.75] } ] },
    T1: { label: "T1 · front legs", hot: ROLE_HEX.motor, parts: [
      { c: [1.35, -2.4, 0], r: [1.7, 1.8, 1.5] }, { c: [-1.35, -2.4, 0], r: [1.7, 1.8, 1.5] } ] },
    T2: { label: "T2 · mid legs + jump", hot: ROLE_HEX.motor, parts: [
      { c: [1.45, -6.0, 0], r: [1.9, 2.1, 1.6] }, { c: [-1.45, -6.0, 0], r: [1.9, 2.1, 1.6] },
      { c: [0, -5.4, -1.1], r: [1.6, 1.4, 0.8], inner: true } ] },     // dorsal flight neuropil
    T3: { label: "T3 · hind legs", hot: ROLE_HEX.motor, parts: [
      { c: [1.25, -9.4, 0], r: [1.6, 1.7, 1.4] }, { c: [-1.25, -9.4, 0], r: [1.6, 1.7, 1.4] } ] },
    ABG: { label: "abdominal ganglion", hot: ROLE_HEX.motor, parts: [
      { c: [0, -12.2, 0], r: [0.9, 1.9, 0.9] } ] },
    CORD: { label: "", hot: ROLE_HEX.motor, parts: [
      { c: [0, -6.6, 0], r: [1.25, 7.0, 1.05], inner: true } ] },
  };
  /* fixed anchors for the faint region captions */
  const REGION_TAGS = [
    { key: "OL_L", at: [7.6, 9.0, 0], text: "optic lobe" },
    { key: "CB", at: [0, 8.9, 0], text: "central brain" },
    { key: "OL_R", at: [-7.6, 9.0, 0], text: "optic lobe" },
    { key: "NECK", at: [2.4, 0.6, 0], text: "neck" },
    { key: "T1", at: [3.9, -2.4, 0], text: "T1" },
    { key: "T2", at: [4.1, -6.0, 0], text: "T2" },
    { key: "T3", at: [3.7, -9.4, 0], text: "T3" },
  ];
  const NECK_PT = new THREE.Vector3(0, 0.6, 0.35);   // brain <-> cord routing point

  /* deterministic per-name hash in [0, 1) */
  function hash(s, k) {
    let h = 2166136261 ^ k;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995); h ^= h >>> 15;
    return (h >>> 0) / 4294967296;
  }
  /* deterministic point inside an ellipsoid (kept off the shell) */
  function scatter(name, c, r, spread = 0.7) {
    const th = hash(name, 1) * Math.PI * 2;
    const ph = Math.acos(2 * hash(name, 2) - 1);
    const rr = spread * (0.35 + 0.65 * Math.cbrt(hash(name, 3)));
    return [c[0] + r[0] * rr * Math.sin(ph) * Math.cos(th),
            c[1] + r[1] * rr * Math.cos(ph),
            c[2] + r[2] * rr * Math.sin(ph) * Math.sin(th)];
  }
  const sideOf = (name, fallbackKey) =>
    /_L$/.test(name) ? 1 : /_R$/.test(name) ? -1 : (hash(name, fallbackKey) < 0.5 ? 1 : -1);
  const SEG = ["T1", "T2", "T3"];

  /* identity -> { region, pos } (schematic anatomy, see header) */
  function placeNeuron(n, roleIdx) {
    const name = String(n.name || n.id);
    const id = name.toUpperCase();
    const sx = sideOf(name, 7);
    let m;
    if ((m = /^TA(\d+)_([PN])(\d+)/.exec(id))) {             // sensory afferent groups
      const slot = +m[1], k = +m[3];
      const s = k === 0 ? 1 : -1;
      const c = [7.0 * s, 7.3 - slot * 0.85, -0.2];
      return { region: s > 0 ? "OL_L" : "OL_R", pos: scatter(name, c, [1.5, 0.55, 1.3], 1) };
    }
    if (/^(GF|DNP01)/.test(id))
      return { region: "CB", pos: scatter(name, [1.6 * sx, 6.3, -1.0], [0.6, 0.5, 0.5], 1) };
    if (/^DNP/.test(id))
      return { region: "CB", pos: scatter(name, [2.6 * sx, 5.4, -1.2], [0.8, 0.7, 0.6], 1) };
    if (/^DNG/.test(id))
      return { region: "GNG", pos: scatter(name, [0.9 * sx, 2.6, 0.5], [0.7, 0.6, 0.7], 1) };
    if (/^PSI/.test(id))
      return { region: "T2", pos: scatter(name, [0.75 * sx, -5.3, -0.6], [0.5, 0.5, 0.4], 1) };
    if (/^(TTM|DLM)/.test(id))
      return { region: "T2", pos: scatter(name, [1.5 * sx, -5.9 - (/^TTM/.test(id) ? 0.8 : 0), -0.6], [0.6, 0.5, 0.5], 1) };
    if ((m = /^LN(\d+)/.exec(id))) {
      const i = +m[1], seg = SEG[i % 3], s = Math.floor(i / 3) % 2 ? -1 : 1;
      const base = REGIONS[seg].parts[s > 0 ? 0 : 1];
      return { region: seg, pos: scatter(name, base.c, base.r, 0.65) };
    }
    if ((m = /^MN(\d+)/.exec(id))) {
      const seg = SEG[+m[1] % 3];
      const base = REGIONS[seg].parts[sx > 0 ? 0 : 1];
      /* leg MNs sit ventro-lateral in their neuropil */
      const c = [base.c[0] + 0.45 * sx, base.c[1], base.c[2] + 0.5];
      return { region: seg, pos: scatter(name, c, base.r, 0.55) };
    }
    /* unknown names (e.g. the synthetic fallback): place by role */
    if (n.role === "sensor") {
      const s = roleIdx % 2 ? -1 : 1;
      return { region: s > 0 ? "OL_L" : "OL_R", pos: scatter(name, [7.0 * s, 5.2, -0.2], [1.8, 2.6, 1.5]) };
    }
    if (n.role === "motor") {
      const seg = SEG[roleIdx % 3];
      const base = REGIONS[seg].parts[roleIdx % 2];
      return { region: seg, pos: scatter(name, base.c, base.r, 0.6) };
    }
    return roleIdx % 3 === 2
      ? { region: "GNG", pos: scatter(name, [0, 2.6, 0.5], [1.6, 0.9, 1.2]) }
      : { region: "CB", pos: scatter(name, [0, 5.5, 0], [3.8, 2.2, 1.9]) };
  }

  function nearestRegion(p) {
    let best = "CB", bd = Infinity;
    for (const [key, reg] of Object.entries(REGIONS)) {
      if (key === "CORD") continue;
      for (const part of reg.parts) {
        const dx = (p[0] - part.c[0]) / part.r[0], dy = (p[1] - part.c[1]) / part.r[1],
          dz = (p[2] - part.c[2]) / part.r[2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bd) { bd = d; best = key; }
      }
    }
    return best;
  }

  /* -------------------------------------------------------- materials --- */
  /* x-ray / confocal look: fresnel rim glow, additive, depth-less */
  function shellMaterial(hot, inner) {
    return new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: {
        base: { value: new THREE.Color(0x5f86c4) },
        hot: { value: new THREE.Color(hot) },
        glow: { value: 0 },
        strength: { value: inner ? 0.45 : 1 },
      },
      vertexShader: `
        varying vec3 vN; varying vec3 vV;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vN = normalize(normalMatrix * normal);
          vV = normalize(-mv.xyz);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform vec3 base, hot; uniform float glow, strength;
        varying vec3 vN; varying vec3 vV;
        void main() {
          float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.3);
          vec3 c = mix(base, hot, clamp(glow * 1.2, 0.0, 1.0));
          float a = (0.05 + f * (0.5 + glow * 0.6)) * strength + glow * 0.05 * strength;
          gl_FragColor = vec4(c * a, 1.0);
        }`,
    });
  }

  let glowTex = null;
  function glowTexture() {
    if (glowTex) return glowTex;
    const cv = document.createElement("canvas");
    cv.width = cv.height = 64;
    const g = cv.getContext("2d");
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(255,255,255,1)");
    grd.addColorStop(0.2, "rgba(255,255,255,0.7)");
    grd.addColorStop(0.5, "rgba(255,255,255,0.15)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    glowTex = new THREE.CanvasTexture(cv);
    return glowTex;
  }
  /* additive point sprites with per-point colour + size (world-scaled) */
  function glowPointsMaterial() {
    return new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { map: { value: glowTexture() }, scale: { value: 300 } },
      vertexShader: `
        attribute float size; attribute vec3 color; varying vec3 vC;
        uniform float scale;
        void main() {
          vC = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * scale / -mv.z;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform sampler2D map; varying vec3 vC;
        void main() {
          vec4 t = texture2D(map, gl_PointCoord);
          gl_FragColor = vec4(vC * t.a, 1.0);
        }`,
    });
  }

  /* ----------------------------------------------------------- state --- */
  const state = {
    inited: false, visible: true,
    renderer: null, scene: null, camera: null, container: null,
    nodes: null, halos: null, nodeData: [], byId: new Map(),
    edges: null, edgeList: [], impulses: null, impulseData: [],
    regionMats: {}, regionGlow: {}, gfAxons: [],
    pulses: [], labels: [], tags: [], counter: null,
    poolMax: { sensor: 0.5, interneuron: 0.5, motor: 0.5 },
    lastPoll: 0, t: 0, cam: null,
  };

  function create(opts) {
    const container = opts.container;
    if (!container || state.inited) return null;

    const w = container.clientWidth || 300;
    const h = container.clientHeight || 300;
    if (getComputedStyle(container).position === "static") container.style.position = "relative";

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    renderer.setClearColor(0x0a0e16, 1);   // opaque: additive shells need a real backdrop
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 200);

    /* --- CNS shells --- */
    const sphere = new THREE.SphereGeometry(1, 40, 28);
    for (const [key, reg] of Object.entries(REGIONS)) {
      const mats = [];
      for (const part of reg.parts) {
        const mat = shellMaterial(reg.hot, part.inner);
        const m = new THREE.Mesh(sphere, mat);
        m.position.set(part.c[0], part.c[1], part.c[2]);
        m.scale.set(part.r[0], part.r[1], part.r[2]);
        m.renderOrder = 0;
        scene.add(m);
        mats.push(mat);
      }
      state.regionMats[key] = mats;
      state.regionGlow[key] = 0;
    }

    /* --- neuron cores: instanced spheres --- */
    const nodes = new THREE.InstancedMesh(new THREE.SphereGeometry(NODE_SIZE, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }), 256);
    nodes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    nodes.renderOrder = 3;
    nodes.frustumCulled = false;
    scene.add(nodes);

    /* --- halos (one glow sprite per neuron) --- */
    const haloGeo = new THREE.BufferGeometry();
    haloGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(256 * 3), 3));
    haloGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(256 * 3), 3));
    haloGeo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(256), 1));
    const halos = new THREE.Points(haloGeo, glowPointsMaterial());
    halos.frustumCulled = false;
    halos.renderOrder = 4;
    scene.add(halos);

    /* --- impulses travelling along active connections --- */
    const impGeo = new THREE.BufferGeometry();
    impGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_IMPULSES * 3), 3));
    impGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(MAX_IMPULSES * 3), 3));
    impGeo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(MAX_IMPULSES), 1));
    const impulses = new THREE.Points(impGeo, glowPointsMaterial());
    impulses.frustumCulled = false;
    impulses.renderOrder = 5;
    scene.add(impulses);
    state.impulseData = new Array(MAX_IMPULSES).fill(0).map(() => ({ t: 0, edge: null }));

    /* --- spike rings (neuron newly among the most active) --- */
    const ringGeo = new THREE.RingGeometry(0.42, 0.55, 24);
    for (let i = 0; i < MAX_PULSES; i++) {
      const m = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide,
        depthWrite: false, blending: THREE.AdditiveBlending }));
      m.visible = false;
      m.renderOrder = 6;
      scene.add(m);
      state.pulses.push({ mesh: m, t: 0, x: 0, y: 0, z: 0 });
    }

    /* --- HTML overlays: region captions, firing labels, counter --- */
    const mk = (css) => {
      const el = document.createElement("div");
      el.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;white-space:nowrap;" +
        "font:10px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;" + css;
      container.appendChild(el);
      return el;
    };
    state.tags = REGION_TAGS.map((t) => ({ ...t, v: new THREE.Vector3(...t.at),
      el: mk("color:rgba(160,185,225,0.55);font-size:9px;letter-spacing:1px;text-transform:uppercase;" +
        "transform:translate(-50%,-50%);") }));
    state.tags.forEach((t) => { t.el.textContent = t.text; });
    state.labels = [0, 1, 2].map(() => mk("padding:2px 6px;border-radius:4px;" +
      "background:rgba(8,11,17,0.78);border:1px solid rgba(255,255,255,0.12);" +
      "transform:translate(10px,-50%);display:none;"));
    state.counter = mk("left:8px;top:auto;bottom:6px;color:#c9d4e3;font-size:10px;");

    /* --- spectate-only drag (no game control; agency contract) --- */
    let dragging = false, lx = 0, ly = 0;
    const cam = { yaw: 0, pitch: 0.12, dragging: () => dragging };
    container.addEventListener("pointerdown", (e) => {
      dragging = true; lx = e.clientX; ly = e.clientY; e.preventDefault();
    });
    window.addEventListener("pointerup", () => { dragging = false; });
    window.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      cam.yaw -= (e.clientX - lx) * 0.008;
      cam.pitch = Math.max(-0.9, Math.min(0.9, cam.pitch + (e.clientY - ly) * 0.006));
      lx = e.clientX; ly = e.clientY;
    });

    function resize() {
      const cw = container.clientWidth, ch = container.clientHeight;
      if (!cw || !ch) return;
      renderer.setSize(cw, ch, false);
      camera.aspect = cw / Math.max(1, ch);
      /* keep the whole CNS (~23 units tall, ~20 wide) in frame */
      const fitH = 25.5, fitW = 21.5;
      const vfov = 2 * Math.atan(Math.max(fitH, fitW / camera.aspect) / (2 * CAM_DIST)) * 180 / Math.PI;
      camera.fov = Math.min(70, vfov);
      camera.updateProjectionMatrix();
      /* point sprites are sized in world units: pixels per unit at depth 1 */
      const ppu = renderer.getDrawingBufferSize(new THREE.Vector2()).y /
        (2 * Math.tan(camera.fov * Math.PI / 360));
      halos.material.uniforms.scale.value = ppu;
      impulses.material.uniforms.scale.value = ppu;
    }
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    Object.assign(state, {
      inited: true, visible: true, renderer, scene, camera, container,
      nodes, halos, impulses, cam,
    });
    resize();

    buildFromNeurons((window.flyBrain && window.flyBrain.getNeurons()) || []);

    return {
      update, render, setVisible,
      /* re-read neurons after a brain re-init / reset */
      rebuild: () => buildFromNeurons((window.flyBrain && window.flyBrain.getNeurons()) || []),
    };
  }

  /* ---------------------------------------------------------- layout --- */
  function schematicLayout() {
    const src = window.flyBrain && window.flyBrain.getDataSource && window.flyBrain.getDataSource();
    return !src || src.synthetic || /ILLUSTRATIVE/i.test(src.notes || "");
  }

  function buildFromNeurons(neurons) {
    const list = neurons.slice(0, 256);
    const schematic = schematicLayout() ||
      !list.every((n) => Array.isArray(n.pos) && n.pos.length === 3 && n.pos.every(Number.isFinite));

    /* real soma coordinates: normalise into the brain's bounding box */
    let norm = null;
    if (!schematic && list.length) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const n of list) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], n.pos[k]); hi[k] = Math.max(hi[k], n.pos[k]); }
      const span = Math.max(hi[0] - lo[0], (hi[1] - lo[1]) * 2.4, 1e-6);
      norm = (p) => [((p[0] - (lo[0] + hi[0]) / 2) / span) * 18,
        5.4 + ((p[1] - (lo[1] + hi[1]) / 2) / span) * 18,
        ((p[2] - (lo[2] + hi[2]) / 2) / span) * 8];
    }

    const roleCount = { sensor: 0, interneuron: 0, motor: 0 };
    const data = [];
    for (const n of list) {
      const role = ROLE_COLORS[n.role] ? n.role : "interneuron";
      const ri = roleCount[role]++;
      let pos, region;
      if (norm) { pos = norm(n.pos); region = nearestRegion(pos); }
      else ({ pos, region } = placeNeuron(n, ri));
      data.push({ id: n.id, name: n.name || n.id, role, region, poolIdx: ri,
        pos: new THREE.Vector3(pos[0], pos[1], pos[2]), bright: 0, target: 0, wasTop: false });
    }
    state.nodeData = data;
    state.byId = new Map(data.map((d) => [d.id, d]));

    /* static instance transforms are rewritten each frame; seed them now */
    const halo = state.halos.geometry.getAttribute("position");
    data.forEach((d, i) => halo.setXYZ(i, d.pos.x, d.pos.y, d.pos.z));
    halo.needsUpdate = true;
    state.halos.geometry.setDrawRange(0, data.length);
    state.nodes.count = data.length;

    buildEdges(neurons);
    buildGiantFibres();
  }

  /* curved connections; brain <-> cord routes through the neck */
  const BRAIN = new Set(["OL_L", "OL_R", "CB", "GNG"]);
  function curvePoint(e, t, out) {
    const u = 1 - t;
    return out.set(
      u * u * e.a.x + 2 * u * t * e.c.x + t * t * e.b.x,
      u * u * e.a.y + 2 * u * t * e.c.y + t * t * e.b.y,
      u * u * e.a.z + 2 * u * t * e.c.z + t * t * e.b.z);
  }
  function buildEdges(neurons) {
    if (state.edges) {
      state.scene.remove(state.edges);
      state.edges.geometry.dispose();
      state.edges = null;
    }
    const all = [];
    for (const n of neurons) {
      const a = state.byId.get(n.id);
      if (!a || !n.connections) continue;
      for (const c of n.connections) {
        const b = state.byId.get(c.target);
        if (b && b !== a) all.push({ src: a, dst: b, w: Math.abs(Number(c.weight) || 0) });
      }
    }
    /* strongest outgoing edge of every neuron, then fill with the strongest overall */
    const chosen = new Set();
    const bestOut = new Map();
    for (const e of all) if (!bestOut.has(e.src) || bestOut.get(e.src).w < e.w) bestOut.set(e.src, e);
    for (const e of bestOut.values()) chosen.add(e);
    for (const e of all.slice().sort((x, y) => y.w - x.w)) {
      if (chosen.size >= MAX_EDGES) break;
      chosen.add(e);
    }
    const edges = Array.from(chosen);
    const wMax = Math.max(1e-6, ...edges.map((e) => e.w));
    for (const e of edges) {
      e.a = e.src.pos; e.b = e.dst.pos; e.wn = e.w / wMax;
      const crosses = BRAIN.has(e.src.region) !== BRAIN.has(e.dst.region);
      if (crosses) e.c = NECK_PT.clone();
      else {
        e.c = e.a.clone().add(e.b).multiplyScalar(0.5);
        e.c.z += 0.9 + e.a.distanceTo(e.b) * 0.12;    // bow toward the viewer
      }
    }
    const pos = new Float32Array(edges.length * CURVE_SEG * 6);
    const col = new Float32Array(edges.length * CURVE_SEG * 6);
    const p = new THREE.Vector3(), q = new THREE.Vector3();
    edges.forEach((e, i) => {
      for (let s = 0; s < CURVE_SEG; s++) {
        curvePoint(e, s / CURVE_SEG, p);
        curvePoint(e, (s + 1) / CURVE_SEG, q);
        const o = (i * CURVE_SEG + s) * 6;
        pos.set([p.x, p.y, p.z, q.x, q.y, q.z], o);
      }
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    lines.renderOrder = 1;
    lines.frustumCulled = false;
    state.scene.add(lines);
    state.edges = lines;
    state.edgeList = edges;
  }

  /* the giant-fibre axons: soma in the brain, descending through the neck to
     their T2 terminals (the escape circuit's signature cells) */
  function buildGiantFibres() {
    for (const g of state.gfAxons) { state.scene.remove(g.mesh); g.mesh.geometry.dispose(); }
    state.gfAxons = [];
    for (const d of state.nodeData) {
      if (!/^GF/i.test(String(d.name))) continue;
      const sx = Math.sign(d.pos.x) || 1;
      const curve = new THREE.CatmullRomCurve3([
        d.pos.clone(),
        new THREE.Vector3(0.9 * sx, 4.2, -0.6),
        new THREE.Vector3(0.35 * sx, 1.6, 0.1),
        new THREE.Vector3(0.3 * sx, -1.0, 0.1),
        new THREE.Vector3(0.55 * sx, -3.8, -0.3),
        new THREE.Vector3(1.1 * sx, -5.6, -0.6),
      ]);
      const mat = new THREE.MeshBasicMaterial({ color: ROLE_HEX.interneuron, transparent: true,
        opacity: 0.18, depthWrite: false, blending: THREE.AdditiveBlending });
      const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.08, 6, false), mat);
      mesh.renderOrder = 2;
      state.scene.add(mesh);
      state.gfAxons.push({ mesh, node: d });
    }
  }

  /* ---------------------------------------------------------- update --- */
  const _dummy = new THREE.Object3D();
  const _col = new THREE.Color();
  const WHITE = new THREE.Color(1, 1, 1);
  const _v = new THREE.Vector3();

  function poll() {
    const fb = window.flyBrain;
    const hid = fb.getHidden();
    const acts = { sensor: hid.sensors || [], interneuron: hid.interneurons || [], motor: hid.motors || [] };
    /* per-pool running max so each layer's hottest neuron reads as hot */
    for (const role of Object.keys(acts)) {
      let m = 0;
      for (const v of acts[role]) m = Math.max(m, Math.abs(Number(v) || 0));
      state.poolMax[role] = Math.max(0.05, m, state.poolMax[role] * 0.97);
    }
    /* ReLU neurons: any activation above 0 means "above threshold" = firing.
       Firing neurons always glow visibly; stronger firing glows brighter. */
    for (const d of state.nodeData) {
      const a = Math.abs(Number(acts[d.role][d.poolIdx]) || 0);
      d.firing = a > 1e-4;
      d.target = d.firing ? 0.45 + 0.55 * Math.min(1, a / state.poolMax[d.role]) : 0;
    }

    /* spike rings for neurons that just entered the top of the ranking */
    const top = fb.getActivity(6);
    const topIds = new Set(top.filter((t) => Math.abs(t.value) > 1e-6).map((t) => t.id));
    for (const d of state.nodeData) {
      const isTop = topIds.has(d.id);
      if (isTop && !d.wasTop) spawnPulse(d.pos, d.role);
      d.wasTop = isTop;
    }

    /* launch impulses along the outgoing connections of firing neurons */
    for (const e of state.edgeList) {
      const b = e.src.target;
      if (b < ACTIVE) continue;
      if (Math.random() < b * (0.25 + 0.55 * e.wn)) spawnImpulse(e);
    }
  }

  function spawnImpulse(e) {
    const slot = state.impulseData.find((s) => s.t <= 0);
    if (!slot) return;
    slot.t = IMPULSE_LIFE * (0.8 + Math.random() * 0.4);
    slot.life = slot.t;
    slot.edge = e;
  }
  function spawnPulse(pos, role) {
    const p = state.pulses.find((x) => x.t <= 0);
    if (!p) return;
    p.t = PULSE_LIFE;
    p.x = pos.x; p.y = pos.y; p.z = pos.z;
    p.mesh.material.color.copy(ROLE_COLORS[role]);
  }

  function update(dt) {
    if (!state.inited) return;
    dt = Math.min(dt || 0.016, 0.1);
    state.t += dt;
    const now = performance.now();
    if (window.flyBrain && now - state.lastPoll > POLL_MS) { state.lastPoll = now; poll(); }

    /* neurons: smoothed brightness -> core size/colour + halo */
    const hc = state.halos.geometry.getAttribute("color");
    const hs = state.halos.geometry.getAttribute("size");
    for (const k in state.regionGlow) state.regionGlow[k] *= Math.max(0, 1 - dt * 2.5);
    let firing = 0;
    for (let i = 0; i < state.nodeData.length; i++) {
      const d = state.nodeData[i];
      d.bright += (d.target - d.bright) * Math.min(1, dt * (d.target > d.bright ? 14 : DECAY));
      if (d.firing) firing++;
      const b = d.bright;
      _dummy.position.copy(d.pos);
      _dummy.scale.setScalar(1 + (HOT_SCALE - 1) * b);
      _dummy.updateMatrix();
      state.nodes.setMatrixAt(i, _dummy.matrix);
      _col.copy(ROLE_COLORS[d.role]).multiplyScalar(0.3 + 0.9 * b);
      if (b > 0.75) _col.lerp(WHITE, (b - 0.75) * 1.6);
      state.nodes.setColorAt(i, _col);
      _col.copy(ROLE_COLORS[d.role]).multiplyScalar(b * b * 1.1 + 0.05);
      hc.setXYZ(i, _col.r, _col.g, _col.b);
      hs.setX(i, 0.6 + b * 2.6);
      state.regionGlow[d.region] = Math.max(state.regionGlow[d.region] || 0, b * b);
    }
    state.nodes.instanceMatrix.needsUpdate = true;
    if (state.nodes.instanceColor) state.nodes.instanceColor.needsUpdate = true;
    hc.needsUpdate = true; hs.needsUpdate = true;

    /* region shells glow with their hottest neuron; the cord with T1-T3 */
    state.regionGlow.CORD = Math.max(state.regionGlow.T1 || 0, state.regionGlow.T2 || 0,
      state.regionGlow.T3 || 0) * 0.6;
    for (const [key, mats] of Object.entries(state.regionMats)) {
      const g = state.regionGlow[key] || 0;
      for (const m of mats) m.uniforms.glow.value += (g - m.uniforms.glow.value) * Math.min(1, dt * 8);
    }

    /* connections brighten with their source neuron */
    if (state.edges) {
      const ec = state.edges.geometry.getAttribute("color");
      const arr = ec.array;
      state.edgeList.forEach((e, i) => {
        const b = e.src.bright;
        const base = 0.014 + 0.026 * e.wn;
        const c = ROLE_COLORS[e.src.role];
        const k = b > ACTIVE ? b * b * (0.25 + 0.45 * e.wn) : 0;
        const r = 0.22 * base + c.r * k, g = 0.3 * base + c.g * k, bl = 0.5 * base + c.b * k;
        for (let s = 0; s < CURVE_SEG * 2; s++) {
          const o = (i * CURVE_SEG * 2 + s) * 3;
          arr[o] = r; arr[o + 1] = g; arr[o + 2] = bl;
        }
      });
      ec.needsUpdate = true;
    }

    /* impulses */
    const ip = state.impulses.geometry.getAttribute("position");
    const ic = state.impulses.geometry.getAttribute("color");
    const is = state.impulses.geometry.getAttribute("size");
    for (let i = 0; i < MAX_IMPULSES; i++) {
      const s = state.impulseData[i];
      if (s.t > 0 && s.edge) {
        s.t -= dt;
        const k = 1 - Math.max(0, s.t) / s.life;
        curvePoint(s.edge, k, _v);
        ip.setXYZ(i, _v.x, _v.y, _v.z);
        const c = ROLE_COLORS[s.edge.src.role];
        const f = Math.sin(Math.PI * Math.min(1, k)) * 0.9 + 0.2;
        ic.setXYZ(i, c.r * f + 0.25 * f, c.g * f + 0.25 * f, c.b * f + 0.25 * f);
        is.setX(i, 0.9);
      } else is.setX(i, 0);
    }
    ip.needsUpdate = true; ic.needsUpdate = true; is.needsUpdate = true;

    /* giant-fibre axons flare with GF */
    for (const g of state.gfAxons) g.mesh.material.opacity = 0.14 + g.node.bright * 0.75;

    /* spike rings */
    for (const p of state.pulses) {
      if (p.t > 0) {
        p.t -= dt;
        const k = 1 - Math.max(0, p.t) / PULSE_LIFE;
        p.mesh.position.set(p.x, p.y, p.z);
        p.mesh.scale.setScalar(1 + k * 2.2);
        p.mesh.material.opacity = (1 - k) * 0.85;
        p.mesh.quaternion.copy(state.camera.quaternion);
        p.mesh.visible = true;
      } else p.mesh.visible = false;
    }

    /* camera: gentle sway around the frontal view unless the user drags */
    const cam = state.cam;
    const yaw = cam.yaw + (cam.dragging() ? 0 : Math.sin(state.t * 0.3) * 0.5);
    state.camera.position.set(
      LOOK_AT.x + Math.sin(yaw) * Math.cos(cam.pitch) * CAM_DIST,
      LOOK_AT.y + Math.sin(cam.pitch) * CAM_DIST,
      LOOK_AT.z + Math.cos(yaw) * Math.cos(cam.pitch) * CAM_DIST);
    state.camera.lookAt(LOOK_AT);

    updateOverlays(firing);
  }

  function project(v, out) {
    const w = state.container.clientWidth, h = state.container.clientHeight;
    _v.copy(v).project(state.camera);
    out.x = (_v.x * 0.5 + 0.5) * w;
    out.y = (-_v.y * 0.5 + 0.5) * h;
    out.vis = _v.z < 1;
    return out;
  }
  const _scr = { x: 0, y: 0, vis: true };
  function updateOverlays(firing) {
    if (!state.visible) return;
    for (const t of state.tags) {
      project(t.v, _scr);
      t.el.style.left = _scr.x + "px";
      t.el.style.top = _scr.y + "px";
      const g = state.regionGlow[t.key] || 0;
      t.el.style.color = g > 0.3 ? "rgba(235,242,255,0.9)" : "rgba(160,185,225,0.5)";
    }
    /* label the three most active neurons that are actually firing */
    const hot = state.nodeData.filter((d) => d.firing && d.bright > ACTIVE)
      .sort((a, b) => b.bright - a.bright).slice(0, 3);
    const placed = [];
    state.labels.forEach((el, i) => {
      const d = hot[i];
      if (!d) { el.style.display = "none"; return; }
      project(d.pos, _scr);
      /* nudge down until it clears the labels already placed */
      let y = _scr.y;
      for (let tries = 0; tries < 6 && placed.some((p) => Math.abs(p.y - y) < 17 && Math.abs(p.x - _scr.x) < 150); tries++) y += 17;
      placed.push({ x: _scr.x, y });
      el.style.display = "";
      el.style.left = _scr.x + "px";
      el.style.top = y + "px";
      el.style.color = "#" + ROLE_COLORS[d.role].getHexString();
      const reg = REGIONS[d.region] ? REGIONS[d.region].label : d.region;
      el.textContent = d.name + " · " + reg;
    });
    state.counter.textContent = "● " + firing + " / " + state.nodeData.length + " neurons firing";
  }

  function render() {
    if (!state.inited || !state.visible) return;
    state.renderer.render(state.scene, state.camera);
  }

  function setVisible(v) {
    state.visible = v;
    if (state.container) state.container.style.display = v ? "" : "none";
  }

  window.flyBrainViz = { create };
})();
