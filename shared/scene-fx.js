/* ============================================================================
 * scene-fx.js — small shared look-and-feel helpers for the three.js games.
 *
 * Purely cosmetic: nothing here touches the fly brain, the simulation, the
 * rewards or the action stream. It exists so the three games share one
 * renderer setup (ACES tone mapping, sRGB output, soft shadows) and a few
 * procedural building blocks (gradient sky dome, additive glow sprites,
 * canvas noise textures, star field) without any image files, add-on
 * modules or bundler — the three.js UMD core only.
 *
 * IIFE exposing global `flyFx`; load after three.js:
 *   flyFx.setupRenderer(renderer, { exposure })
 *   flyFx.skyDome({ top, horizon, bottom, radius, sunDir, sunColor, curve })
 *   flyFx.glowSprite(color, size, opacity)       -> THREE.Sprite (additive)
 *   flyFx.noiseTexture({ base, spread, size, repeat, streaks })
 *   flyFx.stars(count, radius, minY)              -> THREE.Points
 *   flyFx.envFromSky(renderer, skyOpts)           -> PMREM env texture
 *
 * Light intensities: three.js r155+ uses physical light units, so lights
 * written for the old (legacy) units render ~π× too dark. Games pass
 * intensities already scaled for the new units.
 * ========================================================================== */
(function () {
  "use strict";
  if (typeof THREE === "undefined") return;

  function setupRenderer(renderer, opts = {}) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = opts.exposure || 1.0;
    renderer.shadowMap.enabled = opts.shadows !== false;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    return renderer;
  }

  /* Vertical three-stop gradient on an inverted sphere, plus an optional soft
     sun glow. fog:false so the sky stays crisp behind fogged geometry. */
  function skyDome(o = {}) {
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      uniforms: {
        top: { value: new THREE.Color(o.top ?? 0x0b1a3a) },
        horizon: { value: new THREE.Color(o.horizon ?? 0xff8a5c) },
        bottom: { value: new THREE.Color(o.bottom ?? 0x0b0f14) },
        sunDir: { value: (o.sunDir || new THREE.Vector3(0, 0.15, -1)).clone().normalize() },
        sunColor: { value: new THREE.Color(o.sunColor ?? 0xffc477) },
        sunSize: { value: o.sunSize ?? 0.9985 },
        sunGlow: { value: o.sunGlow ?? 0.55 },
        curve: { value: o.curve ?? 0.45 },
      },
      vertexShader: `
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          vec4 p = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * p;
        }`,
      fragmentShader: `
        uniform vec3 top, horizon, bottom, sunColor, sunDir;
        uniform float sunSize, sunGlow, curve;
        varying vec3 vDir;
        void main() {
          float h = vDir.y;
          vec3 c = h > 0.0
            ? mix(horizon, top, pow(clamp(h, 0.0, 1.0), curve))
            : mix(horizon, bottom, pow(clamp(-h * 3.0, 0.0, 1.0), 0.6));
          float d = max(dot(normalize(vDir), sunDir), 0.0);
          c += sunColor * (pow(d, 220.0) * 0.9 + pow(d, 12.0) * 0.25 * sunGlow);
          c += sunColor * smoothstep(sunSize, sunSize + 0.0006, d) * 1.4;
          gl_FragColor = vec4(c, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(o.radius || 900, 32, 16), mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = -1;
    return mesh;
  }

  /* One cached soft radial-gradient texture for every glow sprite. */
  let glowTex = null;
  function glowTexture() {
    if (glowTex) return glowTex;
    const cv = document.createElement("canvas");
    cv.width = cv.height = 128;
    const g = cv.getContext("2d");
    const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, "rgba(255,255,255,1)");
    grd.addColorStop(0.18, "rgba(255,255,255,0.75)");
    grd.addColorStop(0.45, "rgba(255,255,255,0.18)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd;
    g.fillRect(0, 0, 128, 128);
    glowTex = new THREE.CanvasTexture(cv);
    glowTex.colorSpace = THREE.SRGBColorSpace;
    return glowTex;
  }

  /* Additive billboard halo — the cheap stand-in for bloom (the bloom pass
     ships only as an ES module, which this no-bundler repo doesn't load). */
  function glowSprite(color, size, opacity = 1) {
    const mat = new THREE.SpriteMaterial({
      map: glowTexture(), color, transparent: true, opacity,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const s = new THREE.Sprite(mat);
    s.scale.setScalar(size);
    return s;
  }

  /* Seeded canvas grain texture (asphalt, grass, stone). Deterministic so
     every page load looks the same. */
  function noiseTexture(o = {}) {
    const size = o.size || 256;
    const cv = document.createElement("canvas");
    cv.width = cv.height = size;
    const g = cv.getContext("2d");
    const base = new THREE.Color(o.base ?? 0x333333);
    g.fillStyle = "#" + base.getHexString();
    g.fillRect(0, 0, size, size);
    let seed = o.seed || 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const spread = o.spread ?? 0.08;
    const n = size * size * (o.density ?? 0.5);
    for (let i = 0; i < n; i++) {
      const k = (rnd() - 0.5) * 2 * spread;
      const c = base.clone().offsetHSL(0, 0, k);
      g.fillStyle = "#" + c.getHexString();
      const w = 1 + (rnd() < 0.08 ? 1 : 0);
      g.fillRect(rnd() * size, rnd() * size, w, o.streaks ? w * 4 : w);
    }
    if (o.draw) o.draw(g, size);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.anisotropy = 8;
    if (o.repeat) tex.repeat.set(o.repeat[0], o.repeat[1]);
    return tex;
  }

  /* Upper-hemisphere star field (fixed seed). */
  function stars(count = 600, radius = 800, minY = 0.05) {
    const pos = new Float32Array(count * 3);
    let seed = 11;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    for (let i = 0; i < count; i++) {
      const th = rnd() * Math.PI * 2;
      const y = minY + rnd() * (1 - minY);
      const r = Math.sqrt(1 - y * y);
      pos.set([Math.cos(th) * r * radius, y * radius, Math.sin(th) * r * radius], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      color: 0xffffff, size: 1.6, sizeAttenuation: false, transparent: true,
      opacity: 0.8, fog: false, depthWrite: false,
    }));
    pts.frustumCulled = false;
    return pts;
  }

  /* Image-based lighting without addons or image files: render a sky dome
     (same options as skyDome) into a PMREM env map, so metallic/glossy
     PBR surfaces reflect the game's own sky instead of going black. */
  function envFromSky(renderer, o = {}) {
    const pm = new THREE.PMREMGenerator(renderer);
    const sc = new THREE.Scene();
    sc.add(skyDome(Object.assign({ radius: 50 }, o)));
    const tex = pm.fromScene(sc, 0, 0.1, 100).texture;
    pm.dispose();
    return tex;
  }

  window.flyFx = { envFromSky, setupRenderer, skyDome, glowSprite, glowTexture, noiseTexture, stars };
})();
