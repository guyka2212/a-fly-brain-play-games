/* ============================================================================
 * brain-viz.js — live 3D brain panel for the connectome-seeded fly brain.
 *
 * A three.js component shared by all three games. Renders one node per
 * connectome neuron at its stored "pos" (real soma coordinates when the live
 * fetch produced them; otherwise the deterministic illustrative role-shell
 * layout from build_curated.py — sensors outer shell, interneurons mid,
 * motor core), with faint lines for each neuron's strongest connections.
 *
 * Every frame, activations from flyBrain.getHidden() light the nodes up:
 *   - per-node brightness = |activation| (running-smoothed, fast attack, slow decay)
 *   - spikes (new neuron entering the top of getActivity()) emit an expanding
 *     pulse ring at the node, so "which part of the brain is working right now"
 *     is visible at a glance
 *
 * PERFORMANCE: nodes are THREE.InstancedMesh (92 instances of one low-poly
 * sphere, per-instance color via instanceColor), edges are one LineSegments
 * with static geometry. Activation updates touch only instance colors and
 * matrices — no allocations in the hot path. update() is called per rAF; the
 * underlying data pull is throttled to ~20 Hz internally.
 *
 * USAGE (all games):
 *   const viz = flyBrainViz.create({
 *     container: document.getElementById("brain-panel"),  // sized by CSS
 *   });
 *   // then each rAF:
 *   viz.update(dt); viz.render();
 *   // layout toggle handled by the page via viz.setVisible(bool)
 * ========================================================================== */
(function () {
  "use strict";

  if (typeof THREE === "undefined") {
    console.warn("brain-viz: three.js not loaded — brain panel disabled.");
    return;
  }

  const ROLE_COLORS = {
    sensor: new THREE.Color(0xd29922),      // amber
    interneuron: new THREE.Color(0xa371f7), // violet
    motor: new THREE.Color(0x3fb950),       // green
  };

  /* visual constants */
  const NODE_SIZE = 0.42;          // base node radius (world units)
  const HOT_SCALE = 2.1;           // max node scale when fully activated
  const ATTACK = 0.35;             // brightness rise per second fraction
  const DECAY = 1.6;               // brightness fall per second
  const MAX_EDGES = 130;           // strongest connections drawn
  const PULSE_LIFE = 0.6;          // seconds for one spike pulse
  const MAX_PULSES = 24;
  const POLL_MS = 50;              // activation pull rate (~20 Hz)

  const state = {
    inited: false,
    visible: true,
    renderer: null, scene: null, camera: null,
    nodes: null,                    // InstancedMesh
    nodeData: [],                   // {id, name, role, basePos, bright, prevTop}
    edges: null,
    pulses: [],                     // {x,y,z,t}
    pulseMesh: null,
    lastPoll: 0,
    rotT: 0,
    container: null,
    ro: null,
  };

  function create(opts) {
    const container = opts.container;
    if (!container || state.inited) return null;

    const w = container.clientWidth || 300;
    const h = container.clientHeight || 260;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 200);
    camera.position.set(0, 6, 26);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.HemisphereLight(0xbfd4e8, 0x0c0f13, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 0.5);
    sun.position.set(10, 20, 14);
    scene.add(sun);

    /* --- nodes: instanced spheres --- */
    const nodeGeo = new THREE.SphereGeometry(NODE_SIZE, 10, 8);
    const nodeMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const nodes = new THREE.InstancedMesh(nodeGeo, nodeMat, 256);
    nodes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(nodes);

    /* --- build nodeData from the loaded connectome (public flyBrain API) --- */
    const neurons = (window.flyBrain && window.flyBrain.getNeurons()) || [];
    buildFromNeurons(neurons, nodes, scene);

    /* --- spikes: pooled ring sprites --- */
    const pulseGeo = new THREE.RingGeometry(0.55, 0.75, 20);
    const pulseMat = new THREE.MeshBasicMaterial({
      color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide,
      depthWrite: false,
    });
    const pulses = [];
    for (let i = 0; i < MAX_PULSES; i++) {
      const m = new THREE.Mesh(pulseGeo, pulseMat.clone());
      m.visible = false;
      scene.add(m);
      pulses.push({ mesh: m, t: 0 });
    }

    /* --- camera drag (spectate-only, matches the agency contract) --- */
    let dragging = false, lx = 0, ly = 0, yaw = 0.6, pitch = 0.35;
    container.addEventListener("pointerdown", (e) => {
      dragging = true; lx = e.clientX; ly = e.clientY; e.preventDefault();
    });
    window.addEventListener("pointerup", () => { dragging = false; });
    window.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      yaw -= (e.clientX - lx) * 0.008;
      pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - ly) * 0.006));
      lx = e.clientX; ly = e.clientY;
    });

    function resize() {
      const cw = container.clientWidth, ch = container.clientHeight;
      if (!cw || !ch) return;
      renderer.setSize(cw, ch, false);
      camera.aspect = cw / Math.max(1, ch);
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    Object.assign(state, {
      inited: true, visible: true, renderer, scene, camera,
      nodes, edges: null, pulses, lastPoll: 0, rotT: 0, container, ro,
      cam: { yaw, pitch, dragging: () => dragging },
    });

    return {
      update,
      render,
      setVisible,
      /* re-read positions after a brain re-init (e.g. game Restart) */
      rebuild: () => buildFromNeurons(
        (window.flyBrain && window.flyBrain.getNeurons()) || [], nodes, scene),
    };
  }

  function buildFromNeurons(neurons, nodes, scene) {
    const data = [];
    const n = Math.min(neurons.length, 256);
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    /* role fallback layout for neurons without pos (deterministic, mirrors
       build_curated.py's role shells in spirit, compressed into view space) */
    const byRole = { sensor: [], interneuron: [], motor: [] };
    for (const neuron of neurons) if (byRole[neuron.role]) byRole[neuron.role].push(neuron);

    const R = { sensor: 9.5, interneuron: 6.0, motor: 3.2 };
    const YS = { sensor: 1.0, interneuron: 0.65, motor: 0.45 };
    const golden = Math.PI * (3 - Math.sqrt(5));

    let idx = 0;
    for (const role of ["sensor", "interneuron", "motor"]) {
      const list = byRole[role];
      list.forEach((neuron, i) => {
        let p;
        if (Array.isArray(neuron.pos) && neuron.pos.length === 3 &&
            neuron.pos.every(Number.isFinite)) {
          p = neuron.pos.slice();
        } else {
          const y = 1 - (2 * i + 1) / list.length;
          const rr = Math.sqrt(Math.max(0, 1 - y * y));
          const th = golden * i;
          p = [R[role] * rr * Math.cos(th), R[role] * y * YS[role],
               R[role] * rr * Math.sin(th)];
        }
        data.push({
          id: neuron.id, name: neuron.name, role,
          pos: new THREE.Vector3(p[0], p[1], p[2]),
          bright: 0, wasTop: false,
        });
        dummy.position.set(p[0], p[1], p[2]);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        nodes.setMatrixAt(idx, dummy.matrix);
        color.copy(ROLE_COLORS[role]).multiplyScalar(0.25);
        nodes.setColorAt(idx, color);
        idx++;
      });
    }
    nodes.count = idx;
    nodes.instanceMatrix.needsUpdate = true;
    if (nodes.instanceColor) nodes.instanceColor.needsUpdate = true;

    /* --- strongest edges as one static LineSegments --- */
    if (state.edges) {
      scene.remove(state.edges);
      state.edges.geometry.dispose();
      state.edges = null;
    }
    const byId = new Map(data.map((d) => [d.id, d]));
    const edgesAll = [];
    for (const neuron of neurons) {
      const from = byId.get(neuron.id);
      if (!from || !neuron.connections) continue;
      for (const c of neuron.connections) {
        const to = byId.get(c.target);
        if (!to) continue;
        edgesAll.push({ a: from.pos, b: to.pos, w: Math.abs(Number(c.weight) || 0) });
      }
    }
    edgesAll.sort((x, y2) => y2.w - x.w);
    const top = edgesAll.slice(0, MAX_EDGES);
    const linePos = new Float32Array(top.length * 6);
    top.forEach((e, i) => {
      linePos[i * 6] = e.a.x; linePos[i * 6 + 1] = e.a.y; linePos[i * 6 + 2] = e.a.z;
      linePos[i * 6 + 3] = e.b.x; linePos[i * 6 + 4] = e.b.y; linePos[i * 6 + 5] = e.b.z;
    });
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    const lines = new THREE.LineSegments(
      lineGeo,
      new THREE.LineBasicMaterial({ color: 0x3d4a5c, transparent: true, opacity: 0.35 }));
    scene.add(lines);
    state.edges = lines;

    state.nodeData = data;
  }

  /* pull activations + spike info at POLL_MS, update visuals every frame */
  function update(dt) {
    if (!state.inited) return;
    const now = performance.now();

    if (now - state.lastPoll > POLL_MS && window.flyBrain) {
      state.lastPoll = now;

      /* absolute activation magnitudes per pool */
      const hid = window.flyBrain.getHidden();
      const pools = [
        { role: "sensor", acts: hid.sensors },
        { role: "interneuron", acts: hid.interneurons },
        { role: "motor", acts: hid.motors },
      ];
      for (const p of pools) {
        let pi = 0;
        for (const d of state.nodeData) {
          if (d.role === p.role) {
            const target = Math.min(1, Math.abs(Number(p.acts[pi++]) || 0) * 0.8);
            d.target = target;
          }
        }
      }

      /* spikes: neurons newly at the top of the activity ranking */
      const top = window.flyBrain.getActivity(6);
      for (const d of state.nodeData) d.isTop = false;
      for (const t of top) {
        const d = state.nodeData.find((x) => x.id === t.id);
        if (d) {
          d.isTop = true;
          if (!d.wasTop && state.pulses) spawnPulse(d.pos);
          d.wasTop = true;
        }
      }
    }

    /* per-frame smoothing + instance updates */
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    for (let i = 0; i < state.nodeData.length; i++) {
      const d = state.nodeData[i];
      const target = d.target || 0;
      d.bright += (target - d.bright) * Math.min(1, dt * (target > d.bright ? 10 : DECAY));
      const s = 1 + (HOT_SCALE - 1) * d.bright;
      dummy.position.copy(d.pos);
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      state.nodes.setMatrixAt(i, dummy.matrix);
      color.copy(ROLE_COLORS[d.role]).multiplyScalar(0.25 + 0.75 * d.bright);
      state.nodes.setColorAt(i, color);
    }
    state.nodes.instanceMatrix.needsUpdate = true;
    if (state.nodes.instanceColor) state.nodes.instanceColor.needsUpdate = true;

    /* pulse rings */
    for (const p of state.pulses) {
      if (p.t > 0) {
        p.t -= dt;
        const k = 1 - Math.max(0, p.t) / PULSE_LIFE;
        p.mesh.position.set(p.x, p.y, p.z);
        p.mesh.scale.setScalar(1 + k * 4);
        p.mesh.material.opacity = (1 - k) * 0.8;
        p.mesh.lookAt(state.camera.position);
        p.mesh.visible = true;
      } else p.mesh.visible = false;
    }

    /* slow auto-rotation when the user isn't dragging */
    state.rotT += dt;
    if (!state.cam.dragging()) state.cam.yaw += dt * 0.12;
    const cp = state.cam;
    const cx = Math.sin(cp.yaw) * Math.cos(cp.pitch) * 26;
    const cz = Math.cos(cp.yaw) * Math.cos(cp.pitch) * 26;
    const cy = 6 + Math.sin(cp.pitch) * 26;
    state.camera.position.set(cx, cy, cz);
    state.camera.lookAt(0, 0, 0);
  }

  function spawnPulse(pos) {
    const p = state.pulses.find((p) => p.t <= 0);
    if (!p) return;
    p.t = PULSE_LIFE;
    p.x = pos.x; p.y = pos.y; p.z = pos.z;
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
