/**
 * VoidBox — Frontend Application Logic
 * Handles file upload, API calls, and result routing.
 */

'use strict';

// ─── File Handling ─────────────────────────────────────────────────────────────

let currentFileB64 = null;

/**
 * Called when user selects a file from the input.
 */
function handleFile(input) {
    const file = input.files[0];
    if (!file) return;
    loadFile(file);
}

/**
 * Load a File object: show preview, enable run button.
 */
function loadFile(file) {
    if (!file.type.startsWith('image/')) {
        setRunStatus('Only image files are supported (PNG, JPEG, WEBP).', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        currentFileB64 = e.target.result; // data:image/...;base64,...

        const previewWrap = document.getElementById('previewWrap');
        const previewImg = document.getElementById('previewImg');
        const uploadZone = document.getElementById('uploadZone');
        const runBtn = document.getElementById('runBtn');
        const previewMeta = document.getElementById('previewMeta');

        if (previewWrap && previewImg) {
            previewWrap.style.display = 'block';
            previewImg.src = currentFileB64;
            uploadZone.style.display = 'none';
        }

        if (previewMeta) {
            const kb = (file.size / 1024).toFixed(1);
            previewMeta.textContent = `${file.name}  ·  ${kb} KB  ·  ${file.type}`;
        }

        if (runBtn) {
            runBtn.disabled = false;
        }

        setRunStatus('');
    };
    reader.readAsDataURL(file);
}

/**
 * Clear the current image selection.
 */
function clearImage() {
    currentFileB64 = null;

    const previewWrap = document.getElementById('previewWrap');
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const runBtn = document.getElementById('runBtn');

    if (previewWrap) previewWrap.style.display = 'none';
    if (uploadZone) uploadZone.style.display = 'block';
    if (fileInput) fileInput.value = '';
    if (runBtn) runBtn.disabled = true;
    setRunStatus('');
}


// ─── Redaction Call ────────────────────────────────────────────────────────────

/**
 * Read settings from the tool page form, call /api/redact, store results,
 * then navigate to results.html.
 */
async function runRedaction() {
    if (!currentFileB64) {
        setRunStatus('Please upload an image first.', 'error');
        return;
    }

    // Gather settings
    const ocrMode = document.querySelector('input[name="ocr_mode"]:checked')?.value ?? 'smart';
    const payload = {
        image: currentFileB64,
        redact_documents: document.getElementById('chk_documents')?.checked ?? true,
        redact_faces: document.getElementById('chk_faces')?.checked ?? true,
        redact_signatures: document.getElementById('chk_signatures')?.checked ?? true,
        redact_text_fields: document.getElementById('chk_text_fields')?.checked ?? true,
        redact_plates: document.getElementById('chk_plates')?.checked ?? true,
        ocr_mode: ocrMode,
        ocr_confidence: parseFloat(document.getElementById('ocrConf')?.value ?? '0.4'),
        show_detections: document.getElementById('chk_preview')?.checked ?? false,
        fast_mode: document.getElementById('chk_fast')?.checked ?? false,
    };

    // UI: disable button, show spinner
    const runBtn = document.getElementById('runBtn');
    if (runBtn) runBtn.disabled = true;
    setRunStatus('Running... this may take 20–60 seconds.');
    showSpinner('Running detection pipeline...');

    try {
        const resp = await fetch('/api/redact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await resp.json();

        if (!resp.ok || data.error) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }

        // Store a small result id (avoid large base64 in sessionStorage)
        if (data.result_id) {
            sessionStorage.setItem('voidbox_result_id', data.result_id);
            sessionStorage.removeItem('voidbox_result');
            sessionStorage.removeItem('voidbox_original');
        } else {
            sessionStorage.setItem('voidbox_result', JSON.stringify(data));
            sessionStorage.setItem('voidbox_original', currentFileB64);
        }

        // Navigate to results
        window.location.href = 'results.html';

    } catch (err) {
        hideSpinner();
        if (runBtn) runBtn.disabled = false;
        setRunStatus(`Error: ${err.message}`, 'error');
    }
}


// ─── UI Helpers ───────────────────────────────────────────────────────────────

function setRunStatus(msg, type) {
    const el = document.getElementById('runStatus');
    if (!el) return;
    el.textContent = msg;
    el.style.color = type === 'error' ? 'var(--danger)' : 'var(--text-muted)';
}

function showSpinner(text) {
    const overlay = document.getElementById('spinner');
    const textEl = document.getElementById('spinnerText');
    if (!overlay) return;
    if (textEl) textEl.textContent = text;
    overlay.classList.add('visible');
}

function hideSpinner() {
    const overlay = document.getElementById('spinner');
    if (overlay) overlay.classList.remove('visible');
}
