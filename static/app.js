/* ============================================================
   FaceAI — app.js  (Vanilla JS — no framework needed)
   ============================================================ */

const API = '';   // Same origin — empty string → relative path

/* ── Utility ─────────────────────────────────────────────────── */
function toast(msg, type = 'info') {
  const c   = document.getElementById('toast-container');
  const el  = document.createElement('div');
  const icons = {
    success: '✅', error: '❌', info: 'ℹ️',
  };
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || ''}</span> ${msg}`;
  c.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function formatTime() {
  return new Date().toLocaleTimeString('vi-VN', { hour12: false });
}

function showStatus(el, msg, type) {
  el.textContent = msg;
  el.className   = `status-message ${type}`;
  el.hidden      = false;
}

/* ── Tab Navigation ──────────────────────────────────────────── */
const tabs = {
  'nav-webcam':  { tab: 'tab-webcam',   title: 'Webcam Demo Realtime',       sub: 'Nhận diện khuôn mặt trực tiếp qua webcam' },
  'nav-image':   { tab: 'tab-image',    title: 'Nhận diện từ ảnh',           sub: 'Upload hoặc kéo thả ảnh để nhận diện' },
  'nav-persons': { tab: 'tab-persons',  title: 'Quản lý Nhân sự',            sub: 'Đăng ký, tra cứu và xóa nhân sự' },
  'nav-api':     { tab: 'tab-api',      title: 'API Documentation',           sub: 'Tài liệu REST API và WebSocket' },
};

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    const info = tabs[btn.id];
    if (!info) return;

    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    btn.classList.add('active');
    document.getElementById(info.tab).classList.add('active');
    document.getElementById('page-title').textContent   = info.title;
    document.getElementById('page-subtitle').textContent = info.sub;
  });
});

/* ── Health check & status ───────────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    document.getElementById('status-dot').className  = 'status-dot online';
    document.getElementById('status-text').textContent = 'API online';
    document.getElementById('persons-count').textContent = d.persons_in_memory ?? 0;
  } catch {
    document.getElementById('status-dot').className  = 'status-dot offline';
    document.getElementById('status-text').textContent = 'Mất kết nối API';
  }
}
checkHealth();
setInterval(checkHealth, 10000);

/* ════════════════════════════════════════════════════════════════
   WEBCAM TAB — WebSocket realtime recognition
   ════════════════════════════════════════════════════════════════ */
const video      = document.getElementById('local-video');
const canvas     = document.getElementById('overlay-canvas');
const ctx        = canvas.getContext('2d');
const fpsDisplay = document.getElementById('fps-display');
const facesDisp  = document.getElementById('faces-display');
const cameraInactive = document.getElementById('camera-inactive');

let mediaStream  = null;
let ws           = null;
let captureLoop  = null;
let isStreaming  = false;

const btnStart = document.getElementById('btn-start-camera');
const btnStop  = document.getElementById('btn-stop-camera');

btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click',  stopCamera);

document.getElementById('btn-clear-log').addEventListener('click', () => {
  const log = document.getElementById('detection-log');
  log.innerHTML = '<div class="log-empty">Chưa có kết quả nhận diện</div>';
});

document.getElementById('threshold-slider').addEventListener('input', e => {
  document.getElementById('threshold-value').textContent = parseFloat(e.target.value).toFixed(2);
});

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = mediaStream;
    await video.play();

    cameraInactive.classList.add('hidden');
    btnStart.disabled = true;
    btnStop.disabled  = false;
    isStreaming = true;

    connectWebSocket();
  } catch (e) {
    toast('Không thể truy cập camera: ' + e.message, 'error');
  }
}

function stopCamera() {
  isStreaming = false;
  if (captureLoop) { clearInterval(captureLoop); captureLoop = null; }
  if (ws)          { ws.close(); ws = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  video.srcObject  = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cameraInactive.classList.remove('hidden');
  fpsDisplay.textContent  = '-- FPS';
  facesDisp.textContent   = '0 khuôn mặt';
  btnStart.disabled = false;
  btnStop.disabled  = true;
}

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl  = `${proto}://${location.host}/api/recognize/stream`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('WS connected');
    startCapturing();
  };

  ws.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data.error) { console.warn('WS error:', data.error); return; }

      // Draw annotated frame on canvas overlay
      const img = new Image();
      img.onload = () => {
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = 'data:image/jpeg;base64,' + data.image_base64;

      fpsDisplay.textContent = `${data.fps ?? '--'} FPS`;
      facesDisp.textContent  = `${(data.detections || []).length} khuôn mặt`;

      // Log detections
      if (data.detections && data.detections.length > 0) {
        appendDetectionLogs(data.detections);
      }
    } catch {}
  };

  ws.onerror = () => toast('WebSocket lỗi kết nối', 'error');
  ws.onclose = () => {
    if (isStreaming) {
      // Reconnect sau 1s nếu vẫn đang streaming
      setTimeout(() => { if (isStreaming) connectWebSocket(); }, 1000);
    }
  };
}

function startCapturing() {
  const offscreen = document.createElement('canvas');
  const offCtx    = offscreen.getContext('2d');

  captureLoop = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;

    offscreen.width  = video.videoWidth;
    offscreen.height = video.videoHeight;
    offCtx.drawImage(video, 0, 0);

    const threshold   = parseFloat(document.getElementById('threshold-slider').value);
    const applyFilter = document.getElementById('filter-toggle').checked;

    offscreen.toBlob(blob => {
      if (!blob) return;
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result.split(',')[1];
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ frame: b64, threshold, apply_filter: applyFilter }));
        }
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.7);

  }, 200);   // ~5 FPS to server (display stays smooth via canvas)
}

function appendDetectionLogs(detections) {
  const log = document.getElementById('detection-log');
  const empty = log.querySelector('.log-empty');
  if (empty) empty.remove();

  detections.forEach(d => {
    if (!d.recognized) return;   // Only log known persons in sidebar
    const item = document.createElement('div');
    item.className = `log-item ${d.recognized ? 'known' : 'unknown'}`;
    item.innerHTML = `
      <div class="log-name">${d.recognized ? '✅' : '❓'} ${d.name}</div>
      <div class="log-sim">Độ tương đồng: ${(d.similarity * 100).toFixed(1)}%</div>
      <div class="log-time">${formatTime()}</div>
    `;
    log.insertBefore(item, log.firstChild);
    // Keep max 50 log items
    while (log.children.length > 50) log.lastChild.remove();
  });
}

/* ════════════════════════════════════════════════════════════════
   IMAGE TAB — Static image recognition
   ════════════════════════════════════════════════════════════════ */
const dropzone       = document.getElementById('dropzone');
const imageInput     = document.getElementById('image-input');
const previewImg     = document.getElementById('preview-img');
const dropzoneContent = document.getElementById('dropzone-content');
const btnRecognize   = document.getElementById('btn-recognize-image');

dropzone.addEventListener('click', () => imageInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadPreview(file);
});

imageInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) loadPreview(file);
});

document.getElementById('img-threshold').addEventListener('input', e => {
  document.getElementById('img-threshold-value').textContent = parseFloat(e.target.value).toFixed(2);
});

function loadPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  dropzoneContent.style.display = 'none';
  btnRecognize.disabled = false;
  // Store file for submission
  btnRecognize._file = file;
}

btnRecognize.addEventListener('click', async () => {
  const file = btnRecognize._file;
  if (!file) return;

  btnRecognize.disabled = true;
  btnRecognize.innerHTML = '<div class="spinner"></div> Đang xử lý...';

  const threshold   = document.getElementById('img-threshold').value;
  const applyFilter = document.getElementById('img-filter').checked;

  const fd = new FormData();
  fd.append('photo', file);
  fd.append('threshold', threshold);
  fd.append('apply_filter', applyFilter);

  try {
    const r = await fetch(`${API}/api/recognize/image`, { method: 'POST', body: fd });
    const d = await r.json();

    if (!r.ok) throw new Error(d.detail || 'Lỗi API');

    document.getElementById('result-placeholder').style.display = 'none';
    const resultImg = document.getElementById('result-image');
    resultImg.src = 'data:image/jpeg;base64,' + d.image_base64;
    resultImg.hidden = false;

    const list = document.getElementById('img-detections');
    list.innerHTML = '';
    if (d.detections.length === 0) {
      list.innerHTML = '<div style="color:var(--text-dim);font-size:.78rem;text-align:center;padding:12px">Không phát hiện khuôn mặt nào.</div>';
    } else {
      d.detections.forEach(det => {
        const el = document.createElement('div');
        el.className = `detection-item ${det.recognized ? 'known' : 'unknown'}`;
        el.innerHTML = `
          <div>
            <strong>${det.recognized ? '✅ ' + det.name : '❓ Unknown'}</strong>
            <div style="font-size:.72rem;color:var(--text-muted);margin-top:2px">
              Tương đồng: ${(det.similarity * 100).toFixed(1)}% &nbsp;|&nbsp; Conf: ${(det.confidence * 100).toFixed(0)}%
            </div>
          </div>
          <span style="font-size:.7rem;color:var(--text-dim)">ID ${det.person_id ?? '—'}</span>
        `;
        list.appendChild(el);
      });
    }
    toast(`Phát hiện ${d.face_count} khuôn mặt`, 'success');
  } catch (e) {
    toast('Lỗi nhận diện: ' + e.message, 'error');
  } finally {
    btnRecognize.disabled = false;
    btnRecognize.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Nhận diện`;
  }
});

/* ════════════════════════════════════════════════════════════════
   PERSONS TAB — CRUD
   ════════════════════════════════════════════════════════════════ */

// Register photo upload
const regDropzone = document.getElementById('reg-dropzone');
const regPhotoInput = document.getElementById('reg-photo');
const regPreview = document.getElementById('reg-preview');
const regDropzoneContent = document.getElementById('reg-dropzone-content');

regDropzone.addEventListener('click', () => regPhotoInput.click());
regDropzone.addEventListener('dragover', e => { e.preventDefault(); regDropzone.classList.add('dragover'); });
regDropzone.addEventListener('dragleave', () => regDropzone.classList.remove('dragover'));
regDropzone.addEventListener('drop', e => {
  e.preventDefault();
  regDropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadRegPreview(file);
});
regPhotoInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) loadRegPreview(file);
});

function loadRegPreview(file) {
  regPreview.src = URL.createObjectURL(file);
  regPreview.hidden = false;
  regDropzoneContent.style.display = 'none';
  regPhotoInput._file = file;
}

// Register person
document.getElementById('btn-register').addEventListener('click', async () => {
  const name   = document.getElementById('reg-name').value.trim();
  const cccd   = document.getElementById('reg-cccd').value.trim();
  const dob    = document.getElementById('reg-dob').value;
  const gender = document.getElementById('reg-gender').value;
  const phone  = document.getElementById('reg-phone').value.trim();
  const file   = regPhotoInput._file;
  const statusEl = document.getElementById('reg-status');

  if (!name || !cccd) {
    showStatus(statusEl, '⚠️ Họ tên và CCCD là bắt buộc!', 'error');
    return;
  }
  if (!file) {
    showStatus(statusEl, '⚠️ Vui lòng chọn ảnh chân dung!', 'error');
    return;
  }

  const btn = document.getElementById('btn-register');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Đang lưu...';

  const fd = new FormData();
  fd.append('ho_ten', name);
  fd.append('cccd', cccd);
  fd.append('ngay_sinh', dob);
  fd.append('gioi_tinh', gender);
  fd.append('dien_thoai', phone);
  fd.append('photo', file);

  try {
    const r = await fetch(`${API}/api/persons/`, { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Lỗi API');

    showStatus(statusEl, `✅ ${d.message}`, 'success');
    toast(d.message, 'success');
    resetRegForm();
    loadPersons();
    checkHealth();
  } catch (e) {
    showStatus(statusEl, '❌ ' + e.message, 'error');
    toast('Đăng ký thất bại: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v14a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> Lưu nhân sự`;
  }
});

function resetRegForm() {
  document.getElementById('reg-name').value  = '';
  document.getElementById('reg-cccd').value  = '';
  document.getElementById('reg-dob').value   = '';
  document.getElementById('reg-phone').value = '';
  document.getElementById('reg-gender').value = 'Nam';
  regPreview.hidden = true;
  regPreview.src    = '';
  regDropzoneContent.style.display = '';
  regPhotoInput._file = null;
  regPhotoInput.value = '';
}

// Load persons table
async function loadPersons(q = '') {
  const tbody = document.getElementById('persons-tbody');
  tbody.innerHTML = '<tr><td colspan="7" class="table-empty">Đang tải...</td></tr>';

  try {
    const url = q ? `${API}/api/persons/search?q=${encodeURIComponent(q)}` : `${API}/api/persons/`;
    const r   = await fetch(url);
    const persons = await r.json();

    if (persons.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="table-empty">Không có dữ liệu</td></tr>';
      return;
    }

    tbody.innerHTML = persons.map(p => `
      <tr>
        <td>${p.id}</td>
        <td><strong>${p.ho_ten}</strong></td>
        <td>${p.cccd || '—'}</td>
        <td>${p.gioi_tinh || '—'}</td>
        <td>${p.ngay_sinh || '—'}</td>
        <td>${p.dien_thoai || '—'}</td>
        <td>
          <button class="btn-delete-person" data-id="${p.id}" data-name="${p.ho_ten}">Xóa</button>
        </td>
      </tr>
    `).join('');

    // Attach delete handlers
    tbody.querySelectorAll('.btn-delete-person').forEach(btn => {
      btn.addEventListener('click', () => confirmDelete(parseInt(btn.dataset.id), btn.dataset.name));
    });
  } catch {
    tbody.innerHTML = '<tr><td colspan="7" class="table-empty">Lỗi tải dữ liệu</td></tr>';
  }
}

// Search persons
let searchTimeout = null;
document.getElementById('persons-search').addEventListener('input', e => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => loadPersons(e.target.value), 350);
});

document.getElementById('btn-refresh-persons').addEventListener('click', () => {
  document.getElementById('persons-search').value = '';
  loadPersons();
  checkHealth();
});

// Delete confirmation modal
let pendingDeleteId = null;

function confirmDelete(id, name) {
  pendingDeleteId = id;
  document.getElementById('modal-message').textContent = `Bạn có chắc muốn xóa "${name}" (ID: ${id})? Hành động này không thể hoàn tác!`;
  document.getElementById('modal-overlay').hidden = false;
}

document.getElementById('modal-cancel').addEventListener('click', () => {
  document.getElementById('modal-overlay').hidden = true;
  pendingDeleteId = null;
});

document.getElementById('modal-confirm').addEventListener('click', async () => {
  if (!pendingDeleteId) return;
  document.getElementById('modal-overlay').hidden = true;

  try {
    const r = await fetch(`${API}/api/persons/${pendingDeleteId}`, { method: 'DELETE' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Lỗi API');
    toast(d.message, 'success');
    loadPersons();
    checkHealth();
  } catch (e) {
    toast('Xóa thất bại: ' + e.message, 'error');
  } finally {
    pendingDeleteId = null;
  }
});

// Close modal on backdrop click
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) {
    document.getElementById('modal-overlay').hidden = true;
  }
});

/* ── Auto-load persons when switching to that tab ─────────────── */
document.getElementById('nav-persons').addEventListener('click', () => loadPersons());

/* ── Initial load ────────────────────────────────────────────── */
loadPersons();
