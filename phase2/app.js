// ASL fingerspelling — dual-video crossfade player.
// Plays per-letter MP4 clips from ./Asl video data/ with smooth transitions.

const VIDEO_BASE = './Asl%20video%20data/';
const REPEAT_GAP_MS = 300;
const SPACE_MS = 700;

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const urlParams = new URL(window.location.href).searchParams;

// ---------- DOM ----------
const galleryContainer = document.querySelector('.gallery-container');
const wordInput        = document.getElementById('word-input');
const chipBar          = document.getElementById('chip-bar');
const btnPrev          = document.getElementById('btn-prev');
const btnNext          = document.getElementById('btn-next');
const btnPlay          = document.getElementById('btn-play');
const btnRestart       = document.getElementById('btn-restart');
const btnTts           = document.getElementById('btn-tts');
const speedSlider      = document.getElementById('speed-slider');
const currentLetterEl  = document.getElementById('current-letter');

const videoA = document.getElementById('sign-video-a');
const videoB = document.getElementById('sign-video-b');
const videos = [videoA, videoB];
const spaceOverlay = document.getElementById('space-overlay');
const progressFill = document.getElementById('progress-fill');
const speedValue   = document.getElementById('speed-value');
const btnClear     = document.getElementById('btn-clear');
const inputWrapper = document.querySelector('.input-wrapper');
let activeIdx = 0;

// ---------- state ----------
let queue        = [];
let queueIdx     = -1;
let playing      = false;
let pendingTimer = null;
let ttsEnabled   = false;
let playToken    = 0;

// ---------- video helpers ----------
function activeVideo()   { return videos[activeIdx]; }
function inactiveVideo() { return videos[1 - activeIdx]; }

function crossfadeToInactive() {
    videos[activeIdx].classList.remove('active');
    activeIdx = 1 - activeIdx;
    videos[activeIdx].classList.add('active');
}

function showSpaceOverlay() {
    videos.forEach(v => v.classList.remove('active'));
    spaceOverlay.classList.add('visible');
}

function hideSpaceOverlay() {
    spaceOverlay.classList.remove('visible');
}

function loadVideoSrc(v, src) {
    return new Promise((resolve) => {
        let done = false;
        const finish = (ok) => {
            if (done) return;
            done = true;
            v.removeEventListener('canplay', onReady);
            v.removeEventListener('error', onErr);
            resolve(ok);
        };
        const onReady = () => finish(true);
        const onErr   = () => finish(false);
        v.addEventListener('canplay', onReady);
        v.addEventListener('error', onErr);
        v.src = src;
        try { v.load(); } catch {}
    });
}

function applyPlaybackRate() {
    const r = Number(speedSlider.value) || 1;
    videoA.playbackRate = r;
    videoB.playbackRate = r;
    if (speedValue) speedValue.textContent = r.toFixed(1) + '×';
}

function resetProgress() {
    if (progressFill) progressFill.style.width = '0%';
}

function updateProgress() {
    if (!progressFill) return;
    const v = activeVideo();
    if (v && v.duration && isFinite(v.duration) && v.duration > 0) {
        const pct = Math.max(0, Math.min(100, (v.currentTime / v.duration) * 100));
        progressFill.style.width = pct + '%';
    }
}

// ---------- word parsing ----------
function parseWord(txt) {
    const out = [];
    for (const ch of txt.toUpperCase()) {
        if (/[A-Z]/.test(ch)) out.push(ch);
        else if (/\s/.test(ch)) out.push(' ');
    }
    return out;
}

// ---------- gallery / chips / HUD ----------
function initGallery() {
    ALPHABET.forEach(L => {
        const btn = document.createElement('button');
        btn.className = 'letter-card';
        btn.textContent = L;
        btn.dataset.letter = L;
        btn.setAttribute('role', 'tab');
        btn.setAttribute('aria-label', `Sign letter ${L}`);
        btn.addEventListener('click', () => {
            wordInput.value = L;
            playWord(L);
        });
        galleryContainer.appendChild(btn);
    });
}

function highlightGallery(letter) {
    document.querySelectorAll('.letter-card').forEach(c => {
        const is = c.dataset.letter === letter;
        c.classList.toggle('active', is);
        if (is) c.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    });
}

function setCurrentLetter(L) {
    const text = L && L !== ' ' ? L : '–';
    if (currentLetterEl.textContent !== text) {
        currentLetterEl.textContent = text;
        currentLetterEl.classList.remove('pop');
        void currentLetterEl.offsetWidth;
        currentLetterEl.classList.add('pop');
    }
    if (L && L !== ' ') highlightGallery(L);
}

function renderChips() {
    chipBar.innerHTML = '';
    queue.forEach((L, i) => {
        const chip = document.createElement('span');
        const isSpace = L === ' ';
        chip.className = 'chip'
            + (isSpace ? ' space' : '')
            + (i < queueIdx ? ' done' : '')
            + (i === queueIdx ? ' active' : '');
        chip.textContent = isSpace ? '·' : L;
        if (!isSpace) {
            chip.setAttribute('role', 'button');
            chip.setAttribute('tabindex', '0');
            chip.addEventListener('click', () => jumpTo(i));
            chip.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); jumpTo(i); }
            });
        }
        chipBar.appendChild(chip);
    });
}

// ---------- player core ----------
function cancelPending() {
    if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
}

function playWord(word) {
    cancelPending();
    const parsed = parseWord(word);
    if (parsed.length === 0) { stopPlayback(); return; }
    queue = parsed;
    queueIdx = 0;
    playing = true;
    updateUrl();
    renderChips();
    syncPlayButton();
    playToken++;
    playCurrent();
}

function stopPlayback() {
    cancelPending();
    playToken++;
    try { videoA.pause(); } catch {}
    try { videoB.pause(); } catch {}
    hideSpaceOverlay();
    queue = [];
    queueIdx = -1;
    playing = false;
    chipBar.innerHTML = '';
    setCurrentLetter('');
    syncPlayButton();
    updateUrl();
}

async function playCurrent() {
    cancelPending();
    const myToken = ++playToken;

    if (queueIdx < 0 || queueIdx >= queue.length) {
        playing = false;
        syncPlayButton();
        return;
    }
    renderChips();
    const L = queue[queueIdx];

    if (L === ' ') {
        setCurrentLetter('');
        resetProgress();
        showSpaceOverlay();
        const rate = Number(speedSlider.value) || 1;
        pendingTimer = setTimeout(() => {
            if (myToken === playToken) advance();
        }, SPACE_MS / rate);
        return;
    }

    setCurrentLetter(L);
    resetProgress();
    if (ttsEnabled) speakLetter(L);

    const isRepeat = queueIdx > 0 && queue[queueIdx - 1] === L;
    const src = VIDEO_BASE + encodeURIComponent(L) + '.mp4';

    const startPlay = async () => {
        pendingTimer = null;
        const nextV = inactiveVideo();
        const loaded = await loadVideoSrc(nextV, src);
        if (myToken !== playToken) return;
        if (!loaded) {
            console.warn(`[video] failed to load ${L}`);
            advance();
            return;
        }
        applyPlaybackRate();
        try { nextV.currentTime = 0; } catch {}
        try {
            await nextV.play();
        } catch (err) {
            if (myToken !== playToken) return;
            console.warn(`[video] ${L} play failed`, err);
            advance();
            return;
        }
        if (myToken !== playToken) { try { nextV.pause(); } catch {} return; }
        const oldV = activeVideo();
        crossfadeToInactive();
        hideSpaceOverlay();
        try { oldV.pause(); } catch {}
    };

    if (isRepeat) {
        pendingTimer = setTimeout(() => {
            if (myToken === playToken) startPlay();
        }, REPEAT_GAP_MS);
    } else {
        startPlay();
    }
}

function advance() {
    cancelPending();
    if (queueIdx >= queue.length - 1) {
        playing = false;
        syncPlayButton();
        return;
    }
    queueIdx++;
    playCurrent();
}

function jumpTo(idx) {
    if (queue.length === 0) return;
    cancelPending();
    queueIdx = Math.max(0, Math.min(queue.length - 1, idx));
    playing = true;
    syncPlayButton();
    playCurrent();
}

function togglePlay() {
    if (playing) {
        playing = false;
        cancelPending();
        try { activeVideo().pause(); } catch {}
        if ('speechSynthesis' in window) speechSynthesis.cancel();
        syncPlayButton();
        return;
    }

    // Starting playback. If the input has been edited to differ from what's
    // queued, treat the click as a request to play the new text.
    const inputText = wordInput.value.trim();
    const normalizedInput = parseWord(inputText).join('');
    const queueText = queue.join('');
    if (inputText && normalizedInput !== queueText) {
        playWord(inputText);
        return;
    }
    if (queue.length === 0) {
        if (inputText) playWord(inputText);
        return;
    }

    // Resume / restart the current queue.
    playing = true;
    const v = activeVideo();
    const atEnd = queueIdx >= queue.length - 1 && v.ended;
    if (queueIdx < 0 || queueIdx >= queue.length || atEnd) {
        queueIdx = 0;
        playCurrent();
    } else if (v.src && v.readyState >= 2 && !v.ended) {
        const p = v.play();
        if (p && typeof p.catch === 'function') p.catch(() => advance());
    } else {
        playCurrent();
    }
    syncPlayButton();
}

function syncPlayButton() { btnPlay.setAttribute('aria-pressed', String(playing)); }

function speakLetter(L) {
    if (!('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(L);
    u.rate = 1.05;
    u.volume = 0.9;
    speechSynthesis.speak(u);
}

function updateUrl() {
    const w = queue.join('').trim();
    const url = new URL(window.location.href);
    if (w) url.searchParams.set('word', w);
    else url.searchParams.delete('word');
    window.history.replaceState(null, '', url.toString());
}

// ---------- video events ----------
for (const v of videos) {
    v.addEventListener('ended', () => {
        if (playing && v.classList.contains('active')) advance();
    });
    v.addEventListener('error', () => {
        if (playing && v.classList.contains('active')) {
            console.warn('[video] active video error', v.error);
            advance();
        }
    });
    v.addEventListener('timeupdate', () => {
        if (v.classList.contains('active')) updateProgress();
    });
}

// ---------- controls ----------
function wireControls() {
    wordInput.addEventListener('input', (e) => {
        const upper = e.target.value.toUpperCase();
        if (e.target.value !== upper) e.target.value = upper;
        inputWrapper.classList.toggle('has-text', e.target.value.length > 0);
    });
    wordInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            playWord(wordInput.value);
            wordInput.blur();
        }
    });

    btnPlay.addEventListener('click', togglePlay);
    btnPrev.addEventListener('click', () => jumpTo(queueIdx - 1));
    btnNext.addEventListener('click', () => jumpTo(queueIdx + 1));
    btnRestart.addEventListener('click', () => { if (queue.length) jumpTo(0); });
    speedSlider.addEventListener('input', applyPlaybackRate);
    btnTts.addEventListener('click', () => {
        ttsEnabled = !ttsEnabled;
        btnTts.setAttribute('aria-pressed', String(ttsEnabled));
        if (!ttsEnabled && 'speechSynthesis' in window) speechSynthesis.cancel();
    });
    btnClear.addEventListener('click', () => {
        wordInput.value = '';
        inputWrapper.classList.remove('has-text');
        stopPlayback();
        wordInput.focus();
    });

    document.addEventListener('keydown', (e) => {
        const active = document.activeElement;
        const typing = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA');
        if (typing) {
            if (e.key === 'Escape') active.blur();
            return;
        }
        if (e.key === ' ')                    { e.preventDefault(); togglePlay(); }
        else if (e.key === 'ArrowLeft')       { e.preventDefault(); jumpTo(queueIdx - 1); }
        else if (e.key === 'ArrowRight')      { e.preventDefault(); jumpTo(queueIdx + 1); }
        else if (e.key.toLowerCase() === 'r') btnRestart.click();
        else if (e.key.toLowerCase() === 'm') btnTts.click();
        else if (e.key === '/')               { e.preventDefault(); wordInput.focus(); wordInput.select(); }
    });
}

// ---------- boot ----------
function boot() {
    initGallery();
    wireControls();
    applyPlaybackRate();
    const urlWord = urlParams.get('word');
    if (urlWord) {
        wordInput.value = urlWord.toUpperCase();
        inputWrapper.classList.add('has-text');
        playWord(urlWord);
    } else {
        setCurrentLetter('');
        wordInput.focus();
    }
}

boot();
