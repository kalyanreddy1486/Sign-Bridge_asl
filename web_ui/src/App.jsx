import React, { useState, useEffect, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';
import './App.css';

const SOCKET_URL =
  typeof window !== 'undefined' && window.location && window.location.origin
    ? window.location.origin
    : 'http://localhost:5000';

// Browser-side capture settings. Keep resolution modest — MediaPipe handles
// 480p fine and it keeps the uplink bandwidth low for remote deploys.
const CAPTURE_WIDTH = 480;
const CAPTURE_HEIGHT = 360;
const FRAME_INTERVAL_MS = 120; // ~8fps. Leaves headroom for server inference.
const JPEG_QUALITY = 0.7;

const App = () => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [prediction, setPrediction] = useState({
    hand_detected: false,
    letter: null,
    confidence: 0,
    stability: 0,
  });
  const [textBuilder, setTextBuilder] = useState({
    sentence: '',
    stability_progress: 0,
    in_cooldown: false,
    cooldown_remaining: 0,
    letter_added: null,
  });
  const [showAddedToast, setShowAddedToast] = useState(false);
  const [addedLetter, setAddedLetter] = useState('');

  const socketRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const frameTimerRef = useRef(null);
  const toastTimerRef = useRef(null);

  // ---------------------------------------------------------------------
  // Socket lifecycle
  // ---------------------------------------------------------------------
  useEffect(() => {
    const newSocket = io(SOCKET_URL, { reconnection: true });
    setSocket(newSocket);
    socketRef.current = newSocket;

    const handleConnect = () => setConnected(true);
    const handleDisconnect = () => {
      setConnected(false);
    };
    const handleFrameUpdate = (data) => {
      if (data.prediction) setPrediction(data.prediction);
      if (data.text_builder) {
        setTextBuilder(data.text_builder);
        if (data.text_builder.letter_added) {
          setAddedLetter(data.text_builder.letter_added);
          setShowAddedToast(true);
          if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
          toastTimerRef.current = setTimeout(() => {
            setShowAddedToast(false);
            toastTimerRef.current = null;
          }, 1500);
        }
      }
    };
    const handleTextCleared = () => {
      setTextBuilder((prev) => ({ ...prev, sentence: '' }));
    };
    const handleTextUpdated = (data) => {
      setTextBuilder((prev) => ({ ...prev, sentence: data.sentence }));
    };

    newSocket.on('connect', handleConnect);
    newSocket.on('disconnect', handleDisconnect);
    newSocket.on('frame_update', handleFrameUpdate);
    newSocket.on('text_cleared', handleTextCleared);
    newSocket.on('text_updated', handleTextUpdated);

    return () => {
      newSocket.off('connect', handleConnect);
      newSocket.off('disconnect', handleDisconnect);
      newSocket.off('frame_update', handleFrameUpdate);
      newSocket.off('text_cleared', handleTextCleared);
      newSocket.off('text_updated', handleTextUpdated);
      newSocket.close();
      if (toastTimerRef.current) {
        clearTimeout(toastTimerRef.current);
        toastTimerRef.current = null;
      }
    };
  }, []);

  // ---------------------------------------------------------------------
  // Camera + frame pump
  // ---------------------------------------------------------------------
  const stopFramePump = useCallback(() => {
    if (frameTimerRef.current) {
      clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    stopFramePump();
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setCameraStarting(false);
    if (socketRef.current && socketRef.current.connected) {
      socketRef.current.emit('reset_stream');
    }
    setPrediction({ hand_detected: false, letter: null, confidence: 0, stability: 0 });
  }, [stopFramePump]);

  const startFramePump = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = CAPTURE_HEIGHT;
    const ctx = canvas.getContext('2d');

    stopFramePump();
    frameTimerRef.current = setInterval(() => {
      if (!socketRef.current || !socketRef.current.connected) return;
      if (!video.videoWidth || !video.videoHeight) return;

      // Mirror horizontally so what the user sees matches the training data
      // (the original OpenCV pipeline did cv2.flip(frame, 1)).
      ctx.save();
      ctx.translate(CAPTURE_WIDTH, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
      ctx.restore();

      const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
      socketRef.current.emit('video_frame', dataUrl);
    }, FRAME_INTERVAL_MS);
  }, [stopFramePump]);

  const startCamera = useCallback(async () => {
    if (cameraActive || cameraStarting) return;
    setCameraError(null);
    setCameraStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAPTURE_WIDTH },
          height: { ideal: CAPTURE_HEIGHT },
          facingMode: 'user',
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      setCameraActive(true);
      setCameraStarting(false);
      startFramePump();
    } catch (err) {
      console.error('getUserMedia failed:', err);
      setCameraError(err?.message || 'Camera access denied');
      setCameraStarting(false);
      setCameraActive(false);
    }
  }, [cameraActive, cameraStarting, startFramePump]);

  const toggleCamera = useCallback(() => {
    if (cameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  }, [cameraActive, startCamera, stopCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // ---------------------------------------------------------------------
  // Text control events
  // ---------------------------------------------------------------------
  const clearText = useCallback(() => {
    if (socketRef.current) socketRef.current.emit('clear_text');
  }, []);

  const deleteLast = useCallback(() => {
    if (socketRef.current) socketRef.current.emit('delete_last');
  }, []);

  const addSpace = useCallback(() => {
    if (socketRef.current) socketRef.current.emit('add_space');
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      const tag = e.target?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      if (e.key === 'Backspace') deleteLast();
      if (e.key === ' ') addSpace();
      if (e.key === 'Escape') clearText();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteLast, addSpace, clearText]);

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------
  return (
    <div className="page">
      <header id="app-header">
        <div className="wordmark">
          asl <em>detect</em>
        </div>
        <div className="top-meta">sign · read · learn</div>
        <div className={`conn-status ${connected ? 'online' : 'offline'}`}>
          <span className="conn-dot" />
          <span>{connected ? 'connected' : 'offline'}</span>
        </div>
      </header>

      <main id="stage">
        <div className="stage-head">
          <span className="hud-label">Now detecting</span>
          <span className="hud-pill">
            <em>{prediction.letter || '–'}</em>
          </span>
        </div>

        <div id="video-container">
          <video
            ref={videoRef}
            className="video-feed"
            autoPlay
            playsInline
            muted
            style={{
              display: cameraActive ? 'block' : 'none',
              transform: 'scaleX(-1)',
            }}
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {!cameraActive && (
            <div className="video-placeholder">
              <svg viewBox="0 0 24 24" width="56" height="56" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="2" y="6" width="14" height="12" rx="2" />
                <path d="M22 8l-6 4 6 4z" />
              </svg>
              <p>
                {cameraError
                  ? `camera error: ${cameraError}`
                  : cameraStarting
                  ? 'opening camera…'
                  : 'press start to begin detecting'}
              </p>
              {cameraStarting && <div className="spinner" aria-hidden="true" />}
            </div>
          )}

          <div className={`hand-indicator ${prediction.hand_detected ? 'detected' : ''}`}>
            <span className="hand-dot" />
            {prediction.hand_detected ? 'hand detected' : 'no hand'}
          </div>

          <div id="progress-bar" aria-hidden="true">
            <div
              id="progress-fill"
              style={{
                width: `${
                  textBuilder.in_cooldown
                    ? (1 - textBuilder.cooldown_remaining / 1.5) * 100
                    : textBuilder.stability_progress * 100
                }%`,
              }}
            />
          </div>
        </div>

        <div className="metrics-row">
          <div className="metric">
            <span className="metric-label">confidence</span>
            <div className="metric-bar">
              <div className="metric-fill" style={{ width: `${prediction.confidence * 100}%` }} />
            </div>
            <span className="metric-value">
              {(prediction.confidence * 100).toFixed(0)}
              <em>%</em>
            </span>
          </div>
          <div className="metric">
            <span className="metric-label">stability</span>
            <div className="metric-bar">
              <div className="metric-fill" style={{ width: `${prediction.stability * 100}%` }} />
            </div>
            <span className="metric-value">
              {(prediction.stability * 100).toFixed(0)}
              <em>%</em>
            </span>
          </div>
        </div>
      </main>

      <div className="text-card">
        <div className="text-card-head">
          <span className="hud-label">sentence</span>
          <span className="hud-label hint">
            {textBuilder.in_cooldown
              ? `cooldown ${textBuilder.cooldown_remaining.toFixed(1)}s`
              : `hold steady · ${(textBuilder.stability_progress * 100).toFixed(0)}%`}
          </span>
        </div>
        <div className="text-display">
          {textBuilder.sentence || <span className="placeholder">start signing to build text…</span>}
          <span className="cursor">_</span>
        </div>
      </div>

      <footer id="controls-container">
        <div className="nav-btns">
          <button
            className={`btn-play-primary ${cameraActive ? 'is-on' : ''}`}
            onClick={toggleCamera}
            aria-pressed={cameraActive}
            title={cameraActive ? 'Stop camera' : 'Start camera'}
            aria-label={cameraActive ? 'Stop camera' : 'Start camera'}
          >
            {cameraActive ? (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
                <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 5v14l12-7z" fill="currentColor" />
              </svg>
            )}
          </button>
          <button className="icon-btn" onClick={addSpace} title="Add space (Space)" aria-label="Add space">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 10v4M20 10v4M4 14h16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
          <button className="icon-btn" onClick={deleteLast} title="Delete last (Backspace)" aria-label="Delete last letter">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M21 6H8l-5 6 5 6h13a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1zM12 10l4 4M16 10l-4 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button className="icon-btn" onClick={clearText} title="Clear all (Esc)" aria-label="Clear sentence">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M6 6l1 14a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        <a href="/phase2/" className="phase-switch">
          <span className="ps-label">type → sign player</span>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </a>
      </footer>

      <div id="hotkey-hints" aria-hidden="true">
        <span>
          <kbd>space</kbd> add space
        </span>
        <span>
          <kbd>⌫</kbd> delete
        </span>
        <span>
          <kbd>esc</kbd> clear
        </span>
        <span className="hint-meta">hold sign 0.8s to add · no hand 1.2s = space</span>
      </div>

      {showAddedToast && (
        <div className="toast">
          <span className="toast-icon">✓</span>
          <span className="toast-content">
            added <em>{addedLetter}</em>
          </span>
        </div>
      )}
    </div>
  );
};

export default App;
