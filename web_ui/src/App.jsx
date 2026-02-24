import React, { useState, useEffect, useCallback } from 'react';
import { io } from 'socket.io-client';
import './App.css';

const App = () => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [frame, setFrame] = useState(null);
  const [prediction, setPrediction] = useState({
    hand_detected: false,
    letter: null,
    confidence: 0,
    stability: 0
  });
  const [textBuilder, setTextBuilder] = useState({
    sentence: '',
    stability_progress: 0,
    in_cooldown: false,
    cooldown_remaining: 0,
    letter_added: null
  });
  const [showAddedToast, setShowAddedToast] = useState(false);
  const [addedLetter, setAddedLetter] = useState('');

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      setConnected(true);
      console.log('Connected to server');
    });

    newSocket.on('disconnect', () => {
      setConnected(false);
      setCameraActive(false);
      console.log('Disconnected from server');
    });

    newSocket.on('frame_update', (data) => {
      setFrame(data.frame);
      setPrediction(data.prediction);
      setTextBuilder(data.text_builder);
      
      // Show toast when letter is added
      if (data.text_builder.letter_added) {
        setAddedLetter(data.text_builder.letter_added);
        setShowAddedToast(true);
        setTimeout(() => setShowAddedToast(false), 1500);
      }
    });

    newSocket.on('camera_status', (data) => {
      if (data.status === 'started') {
        setCameraActive(true);
      } else if (data.status === 'stopped') {
        setCameraActive(false);
        setFrame(null);
      }
    });

    newSocket.on('text_cleared', () => {
      setTextBuilder(prev => ({ ...prev, sentence: '' }));
    });

    newSocket.on('text_updated', (data) => {
      setTextBuilder(prev => ({ ...prev, sentence: data.sentence }));
    });

    return () => newSocket.close();
  }, []);

  const toggleCamera = useCallback(() => {
    if (!socket) return;
    
    if (cameraActive) {
      socket.emit('stop_camera');
    } else {
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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Backspace') deleteLast();
      if (e.key === ' ') addSpace();
      if (e.key === 'Escape') clearText();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteLast, addSpace, clearText]);

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'var(--success)';
    if (conf >= 0.6) return 'var(--warning)';
    return 'var(--danger)';
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">✋</div>
          <h1>ASL Recognition</h1>
        </div>
        <div className="status-bar">
          <div className={`status-dot ${connected ? 'online' : 'offline'}`} />
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Left Panel - Camera */}
        <div className="camera-panel">
          <div className="panel-header">
            <h2>Camera Feed</h2>
            <button 
              className={`camera-btn ${cameraActive ? 'active' : ''}`}
              onClick={toggleCamera}
            >
              {cameraActive ? '⏹ Stop' : '▶ Start'} Camera
            </button>
          </div>
          
          <div className="video-container">
            {frame ? (
              <img 
                src={frame} 
                alt="Camera feed" 
                className="video-feed animate-scale-in"
              />
            ) : (
              <div className="video-placeholder">
                <div className="placeholder-icon">📹</div>
                <p>{cameraActive ? 'Loading camera...' : 'Click Start Camera to begin'}</p>
              </div>
            )}
            
            {/* Hand detection indicator */}
            <div className={`hand-indicator ${prediction.hand_detected ? 'detected' : ''}`}>
              {prediction.hand_detected ? '✋ Hand Detected' : '✋ No Hand'}
            </div>
          </div>
        </div>

        {/* Right Panel - Prediction & Text Builder */}
        <div className="info-panel">
          {/* Prediction Card */}
          <div className="card prediction-card">
            <h3>Live Prediction</h3>
            
            <div className="prediction-display">
              <div 
                className="predicted-letter"
                style={{ 
                  color: prediction.letter ? getConfidenceColor(prediction.confidence) : 'var(--gray)'
                }}
              >
                {prediction.letter || '-'}
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="metric-row">
              <span className="metric-label">Confidence</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ 
                    width: `${prediction.confidence * 100}%`,
                    background: getConfidenceColor(prediction.confidence)
                  }}
                />
              </div>
              <span className="metric-value">{(prediction.confidence * 100).toFixed(0)}%</span>
            </div>

            {/* Stability Bar */}
            <div className="metric-row">
              <span className="metric-label">Stability</span>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ 
                    width: `${prediction.stability * 100}%`,
                    background: 'var(--accent)'
                  }}
                />
              </div>
              <span className="metric-value">{(prediction.stability * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Text Builder Card */}
          <div className="card text-builder-card">
            <h3>Text Builder</h3>
            
            {/* Progress Bar */}
            <div className="builder-progress">
              <div className="progress-header">
                <span>Hold to confirm</span>
                <span className="progress-percent">
                  {textBuilder.in_cooldown 
                    ? `Cooldown: ${textBuilder.cooldown_remaining.toFixed(1)}s`
                    : `${(textBuilder.stability_progress * 100).toFixed(0)}%`
                  }
                </span>
              </div>
              <div className="progress-bar large">
                <div 
                  className={`progress-fill ${textBuilder.in_cooldown ? 'cooldown' : ''}`}
                  style={{ width: `${textBuilder.in_cooldown 
                    ? (1 - textBuilder.cooldown_remaining / 1.5) * 100 
                    : textBuilder.stability_progress * 100}%` 
                  }}
                />
              </div>
            </div>

            {/* Text Display */}
            <div className="text-display">
              <div className="text-content">
                {textBuilder.sentence || <span className="placeholder">Start signing to build text...</span>}
                <span className="cursor">_</span>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="control-buttons">
              <button className="btn btn-secondary" onClick={addSpace}>
                <span>␣</span> Space
              </button>
              <button className="btn btn-warning" onClick={deleteLast}>
                <span>⌫</span> Delete
              </button>
              <button className="btn btn-danger" onClick={clearText}>
                <span>✕</span> Clear
              </button>
            </div>

            <div className="keyboard-hint">
              Shortcuts: Space = Space | Backspace = Delete | Esc = Clear
            </div>
          </div>
        </div>
      </main>

      {/* Toast Notification */}
      {showAddedToast && (
        <div className="toast animate-slide-in">
          <div className="toast-icon">✓</div>
          <div className="toast-content">
            <span className="toast-title">Letter Added</span>
            <span className="toast-letter">{addedLetter}</span>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>Hold sign steady for 0.8s to add • No hand for 1.2s = Space</p>
      </footer>
    </div>
  );
};

export default App;
