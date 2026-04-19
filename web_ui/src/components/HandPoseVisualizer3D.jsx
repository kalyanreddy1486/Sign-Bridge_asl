import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ─── Hand skeleton connections ───────────────────────────────────────────────
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const JOINT_COUNT = 21;
const FINGERTIP_INDICES = new Set([4, 8, 12, 16, 20]);

// ─── REST POSE: flat open palm (21 landmarks × 3 floats, wrist-relative) ─────
const REST_POSE = [
  0.000,  0.000,  0.000,   // 0  wrist
 -0.060, -0.070,  0.020,   // 1  thumb CMC
 -0.120, -0.140,  0.035,   // 2  thumb MCP
 -0.170, -0.200,  0.040,   // 3  thumb IP
 -0.210, -0.260,  0.045,   // 4  thumb tip
 -0.040, -0.290,  0.005,   // 5  index MCP
 -0.050, -0.400,  0.005,   // 6  index PIP
 -0.055, -0.480,  0.005,   // 7  index DIP
 -0.058, -0.550,  0.005,   // 8  index tip
  0.005, -0.300,  0.000,   // 9  middle MCP
  0.005, -0.420,  0.000,   // 10 middle PIP
  0.005, -0.510,  0.000,   // 11 middle DIP
  0.005, -0.580,  0.000,   // 12 middle tip
  0.050, -0.290, -0.005,   // 13 ring MCP
  0.055, -0.400, -0.005,   // 14 ring PIP
  0.058, -0.475, -0.005,   // 15 ring DIP
  0.060, -0.540, -0.005,   // 16 ring tip
  0.095, -0.265, -0.015,   // 17 pinky MCP
  0.105, -0.360, -0.015,   // 18 pinky PIP
  0.110, -0.425, -0.015,   // 19 pinky DIP
  0.115, -0.480, -0.015,   // 20 pinky tip
];

// ─── Convert 63-float array → array of THREE.Vector3 ─────────────────────────
const SCALE = 3.5;
const Y_OFFSET = 1.0; // push hand upward so it's centred in view

function poseToPoints(flat) {
  if (!Array.isArray(flat) || flat.length < JOINT_COUNT * 3) return null;
  const pts = [];
  for (let i = 0; i < JOINT_COUNT; i++) {
    pts.push(new THREE.Vector3(
       flat[i * 3]     * SCALE,
      -flat[i * 3 + 1] * SCALE + Y_OFFSET,   // flip Y
       flat[i * 3 + 2] * SCALE * 0.3,
    ));
  }
  return pts;
}

const restPoints = poseToPoints(REST_POSE);

// ─── Component ────────────────────────────────────────────────────────────────
const HandPoseVisualizer3D = ({ pose, currentLetter, playing, isRest }) => {
  const containerRef = useRef(null);
  const stateRef     = useRef(null);   // all Three.js objects in one ref
  const rafRef       = useRef(null);
  const playingRef   = useRef(playing);

  // current / target positions for lerp
  const curPts = useRef(restPoints.map(p => p.clone()));
  const tgtPts = useRef(restPoints.map(p => p.clone()));

  // ── Scene setup (once) ──────────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const W = el.clientWidth  || 600;
    const H = el.clientHeight || 380;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.setClearColor(0x0d1220, 1);
    el.innerHTML = '';
    el.appendChild(renderer.domElement);

    // Scene
    const scene = new THREE.Scene();

    // Camera – positioned to see range roughly [-1,1] in X/Y
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
    camera.position.set(0, 1, 7);
    camera.lookAt(0, 1, 0);

    // ── OrbitControls ──
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping  = true;
    controls.dampingFactor  = 0.08;
    controls.target.set(0, 1, 0);
    controls.minDistance = 3;
    controls.maxDistance = 15;
    controls.enablePan = false;
    controls.update();

    // ── Lighting ──
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0xfff5e6, 1.0);
    key.position.set(3, 5, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x7dd3fc, 0.5);
    rim.position.set(-3, 2, -4);
    scene.add(rim);

    // ── Grid floor ──
    const grid = new THREE.GridHelper(8, 16, 0x334155, 0x1e293b);
    grid.position.y = -0.5;
    scene.add(grid);

    // ── Joint spheres ──
    const jointMeshes = [];
    for (let i = 0; i < JOINT_COUNT; i++) {
      const isTip   = FINGERTIP_INDICES.has(i);
      const isWrist = i === 0;
      const r = isWrist ? 0.13 : isTip ? 0.11 : 0.08;
      const color = isWrist ? 0xb07030 : isTip ? 0xd4956b : 0xc68642;
      const mat  = new THREE.MeshPhongMaterial({ color, specular: 0x220000, shininess: 35 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 14), mat);
      mesh.position.copy(curPts.current[i]);
      scene.add(mesh);
      jointMeshes.push(mesh);
    }

    // ── Bone lines (reliable fallback) ──
    const bonePts  = [];
    for (const [a, b] of HAND_CONNECTIONS) {
      bonePts.push(curPts.current[a].clone(), curPts.current[b].clone());
    }
    const boneGeo = new THREE.BufferGeometry().setFromPoints(bonePts);
    const boneMat = new THREE.LineBasicMaterial({ color: 0xd4956b, linewidth: 2 });
    const boneLines = new THREE.LineSegments(boneGeo, boneMat);
    scene.add(boneLines);

    // ── Resize ──
    const onResize = () => {
      const w = el.clientWidth  || 600;
      const h = el.clientHeight || 380;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(el);

    // ── Animation loop ──
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);

      // Lerp current → target
      for (let i = 0; i < JOINT_COUNT; i++) {
        curPts.current[i].lerp(tgtPts.current[i], 0.10);
      }

      // Update joint positions
      for (let i = 0; i < JOINT_COUNT; i++) {
        jointMeshes[i].position.copy(curPts.current[i]);
      }

      // Update bone line positions
      const posArr = boneGeo.attributes.position.array;
      let off = 0;
      for (const [a, b] of HAND_CONNECTIONS) {
        const pa = curPts.current[a];
        const pb = curPts.current[b];
        posArr[off++] = pa.x; posArr[off++] = pa.y; posArr[off++] = pa.z;
        posArr[off++] = pb.x; posArr[off++] = pb.y; posArr[off++] = pb.z;
      }
      boneGeo.attributes.position.needsUpdate = true;

      // Idle rotation
      if (!playingRef.current) scene.rotation.y += 0.003;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    stateRef.current = { renderer, scene, controls, boneGeo, boneMat, ro };

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      controls.dispose();
      boneGeo.dispose();
      boneMat.dispose();
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Update target when pose / isRest changes ────────────────────────────────
  useEffect(() => {
    const newPts = (isRest || !pose) ? restPoints : poseToPoints(pose);
    if (!newPts) return;
    for (let i = 0; i < JOINT_COUNT; i++) {
      tgtPts.current[i].copy(newPts[i]);
    }
  }, [pose, isRest]);

  // ── Sync playingRef ─────────────────────────────────────────────────────────
  useEffect(() => {
    playingRef.current = playing;
    if (playing && stateRef.current?.scene) {
      stateRef.current.scene.rotation.y = 0;
    }
  }, [playing]);

  return (
    <div className="phase2-visualizer-wrap">
      <div className="phase2-visualizer-header">
        <span>3D Hand Pose</span>
        <span className={`phase2-pill ${playing ? 'playing' : ''}`}>
          {playing ? `▶ ${currentLetter || '…'}` : 'Idle — drag to rotate'}
        </span>
      </div>
      <div style={{ position: 'relative' }}>
        <div
          ref={containerRef}
          className="phase2-canvas"
          style={{ height: '400px', width: '100%' }}
        />
        {currentLetter && playing && (
          <div className="canvas-letter-overlay">{currentLetter}</div>
        )}
      </div>
    </div>
  );
};

export default HandPoseVisualizer3D;
