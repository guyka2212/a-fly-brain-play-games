/* ============================================================================
 * assets-loader.js — GLB loader for the Blender-built game meshes.
 *
 * Loads one of shared/assets/*.glb (self-contained binary glTF, geometry +
 * PBR materials embedded, produced by tools/blender-assets/build_models.py)
 * with THREE.GLTFLoader and returns a THREE.Group. On ANY failure (offline,
 * missing file, parse error) it calls onError() and the game keeps its
 * primitive-procedural model — the page always runs.
 *
 * IIFE exposing global `flyAssets`; load after three.js (GLTFLoader ships in
 * the three.js examples — we inline a tiny GLB parser instead so no extra CDN
 * file is needed; see parseGlb below).
 *
 * API:
 *   flyAssets.load("car").then(group => {...}, err => {...})
 *   // resolved group has .userData.tris for logging
 * ========================================================================== */
(function () {
  "use strict";

  const BASE = "../shared/assets/";

  /* Tiny binary-glTF parser: exactly what our builder emits — one mesh of
     N primitives with POSITION + NORMAL accessors and ushort indices, plus
     pbrMetallicRoughness materials (kept as MeshStandardMaterial, named after
     the glTF material so games can tint e.g. only the body). ~90 lines instead of the 300 KB loader. */
  function hasNormals(nor) {
    for (let i = 0; i < nor.length; i++) if (nor[i] !== 0) return true;
    return false;
  }

  function parseGlb(buffer) {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== 0x46546c67) throw new Error("not glTF");
    const jsonLen = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen)));
    const binOff = 20 + jsonLen + 8;
    const accType = { VEC3: 3, SCALAR: 1 };
    const accSize = { 5126: 4, 5123: 2 };

    function readAcc(i) {
      const a = json.accessors[i];
      const v = json.bufferViews[a.bufferView];
      const off = binOff + (v.byteOffset || 0);
      const n = accType[a.type] * a.count;
      if (a.componentType === 5126) {
        const f32 = new Float32Array(a.count * accType[a.type]);
        for (let k = 0; k < n; k++) f32[k] = view.getFloat32(off + k * 4, true);
        return f32;
      }
      const u16 = new Uint16Array(a.count);
      for (let k = 0; k < a.count; k++) u16[k] = view.getUint16(off + k * 2, true);
      return u16;
    }

    const group = new THREE.Group();
    let tris = 0;
    for (const prim of json.meshes[0].primitives) {
      const pos = readAcc(prim.attributes.POSITION);
      const nor = readAcc(prim.attributes.NORMAL);
      const idx = readAcc(prim.indices);
      let geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
      /* Older builder runs exported all-zero normals (NaN after normalize ->
         black lit surfaces). Rebuild flat normals then — matching Blender's
         default flat-shaded primitives. */
      if (!hasNormals(nor)) {
        geo = geo.toNonIndexed();
        geo.computeVertexNormals();
      }
      const jm = json.materials[prim.material];
      const m = jm.pbrMetallicRoughness;
      const c = m.baseColorFactor || [1, 1, 1, 1];
      /* glTF base colours are linear, which is what THREE.Color(r, g, b)
         takes. Lamp/eye materials glow so they read at night. */
      const glow = /light|eye/.test(jm.name || "");
      const color = new THREE.Color(c[0], c[1], c[2]);
      const mat = new THREE.MeshStandardMaterial({
        color,
        metalness: m.metallicFactor ?? 0.2,
        roughness: m.roughnessFactor ?? 0.5,
        emissive: glow ? color : 0x000000,
        emissiveIntensity: glow ? 1.6 : 0,
        transparent: c[3] < 1,
        opacity: c[3],
      });
      mat.name = jm.name || "";
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = mat.name;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      tris += geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3;
    }
    group.userData.tris = tris;
    return group;
  }

  async function load(name) {
    const res = await fetch(BASE + name + ".glb");
    if (!res.ok) throw new Error("asset fetch " + name + " -> " + res.status);
    const buf = await res.arrayBuffer();
    const group = parseGlb(buf);
    console.log("flyAssets: loaded " + name + ".glb (" +
      group.userData.tris + " tris, Blender-built)");
    return group;
  }

  window.flyAssets = { load };
})();
