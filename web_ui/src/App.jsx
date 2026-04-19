import React, { useState, useEffect, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';
import './App.css';

const SOCKET_URL =
  typeof window !== 'undefined' && window.location && window.location.origin
    ? window.location.origin
    : 'http://localhost:5000';

const App = () => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [frame, setFrame] = useState(null);
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
  const toastTimerRef = useRef(null);

  useEffect(() => {
    const newSocket = io(SOCKET_URL, { reconnection: true });
    setSocket(newSocket);

    const handleConnect = () => setConnected(true);
    const handleDisconnect = () => {
      setConnected(false);
      setCameraActive(false);
      setCameraStarting(false);
    };
    const handleFrameUpdate = (data) => {
      setFrame(data.frame);
      setPrediction(data.prediction);
      setTextBuilder(data.text_builder);
      setCameraStarting(false);

      if (data.text_builder.letter_added) {
        setAddedLetter(data.text_builder.letter_added);
        setShowAddedToast(true);
        if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
        toastTimerRef.current = setTimeout(() => {
          setShowAddedToast(false);
          toastTimerRef.current = null;
        }, 1500);
      }
    };
    const handleCameraStatus = (data) => {
      if (data.status === 'started') {
        setCameraActive(true);
      } else if (data.status === 'stopped') {
        setCameraActive(false);
        setCameraStarting(false);
        setFrame(null);
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
    newSocket.on('camera_status', handleCameraStatus);
    newSocket.on('text_cleared', handleTextCleared);
    newSocket.on('text_updated', handleTextUpdated);

    return () => {
      newSocket.off('connect', handleConnect);
      newSocket.off('disconnect', handleDisconnect);
      newSocket.off('frame_update', handleFrameUpdate);
      newSocket.off('camera_status', handleCameraStatus);
      newSocket.off('text_cleared', handleTextCleared);
      newSocket.off('text_updated', handleTextUpdated);
      newSocket.close();
      if (toastTimerRef.current) {
        clearTimeout(toastTimerRef.current);
        toastTimerRef.current = null;
      }
    };
  }, []);

  const toggleCamera = useCallback(() => {
    if (!socket) return;
    if (cameraActive) {
      socket.emit('stop_camera');
    } else {
      setCameraStarting(true);
      socket.emit('start_camera');
    }
  }, [socket, cameraActive]);

  const clearText = useCallback(() => {
    if (socket) socket.emit('clear_text');
  }, [socket]);

  const deleteLast = useCallback(() => {
    if (socket) socket.emit('delete_last');
  }, [socket]);

  const addSpace = useCallback(() => {
    if (socket) socket.emit('add_space');
  }, [socket]);

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
          {frame ? (
            <img src={frame} alt="Camera feed" className="video-feed" />
          ) : (
            <div className="video-placeholder">
              <svg viewBox="0 0 24 24" width="56" height="56" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="2" y="6" width="14" height="12" rx="2" />
                <path d="M22 8l-6 4 6 4z" />
              </svg>
              <p>
                {cameraStarting
                  ? 'opening camera…'
                  : cameraActive
                  ? 'warming up…'
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
              <div
                className="metric-fill"
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
            <span className="metric-value">
              {(prediction.confidence * 100).toFixed(0)}
              <em>%</em>
            </span>
          </div>
          <div className="metric">
            <span className="metric-label">stability</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: `${prediction.stability * 100}%` }}
              />
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
          {textBuilder.sentence || (
            <span className="placeholder">start signing to build text…</span>
          )}
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
