const form = document.getElementById('upload-form');
const statusEl = document.getElementById('status');
const videoWrap = document.getElementById('video-wrapper');
const downloadLink = document.getElementById('download-link');
const fileInput = document.getElementById('file-input');
const thumbWrap = document.getElementById('thumb-wrap');
const progressBar = document.getElementById('upload-progress-bar');
const progressContainer = document.getElementById('upload-progress');
const spinner = document.getElementById('spinner');

// Show client-side thumbnail preview when a file is selected
fileInput.addEventListener('change', (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) {
    thumbWrap.innerHTML = 'No file selected';
    return;
  }
  const url = URL.createObjectURL(f);
  thumbWrap.innerHTML = '';
  const vid = document.createElement('video');
  vid.src = url;
  vid.muted = true;
  vid.playsInline = true;
  vid.controls = false;
  vid.autoplay = true;
  vid.loop = true;
  vid.width = 320;
  thumbWrap.appendChild(vid);
});

form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  const f = fileInput.files && fileInput.files[0];
  if (!f) {
    statusEl.textContent = 'Select a video first';
    return;
  }

  statusEl.textContent = 'Uploading...';
  videoWrap.innerHTML = '';
  downloadLink.style.display = 'none';
  progressContainer.style.display = 'block';
  progressBar.style.width = '0%';
  spinner.style.display = 'block';

  // Use fetch and get back a small JSON payload with the URL to the processed file.
  // Then set the returned URL as the video src so the browser can stream it normally
  // (and handle range requests / progressive playback).
  const fd = new FormData();
  fd.append('video', f);

  // Show a simple indeterminate progress while server is processing
  progressBar.style.width = '10%';

  fetch('/process', { method: 'POST', body: fd })
    .then(async (res) => {
      spinner.style.display = 'none';
      progressContainer.style.display = 'none';
      if (!res.ok) {
        const text = await res.text();
        statusEl.textContent = 'Server error: ' + text;
        return;
      }
      const data = await res.json();
      if (!data || !data.url) {
        statusEl.textContent = 'Invalid server response';
        return;
      }

      const videoUrl = data.url;
      const outVideo = document.createElement('video');
      outVideo.controls = true;
      outVideo.src = videoUrl;
      outVideo.width = Math.min(900, window.innerWidth - 200);
      outVideo.muted = true; // allow autoplay in most browsers
      outVideo.autoplay = true;
      videoWrap.innerHTML = '';
      videoWrap.appendChild(outVideo);

      // try to play immediately (may be blocked if not muted)
      outVideo.play().catch(() => {});

      downloadLink.href = videoUrl;
      downloadLink.style.display = 'inline-block';
      downloadLink.download = 'annotated_output.mp4';
      downloadLink.textContent = 'Download annotated video';
      statusEl.textContent = 'Completed';
    })
    .catch((err) => {
      spinner.style.display = 'none';
      progressContainer.style.display = 'none';
      statusEl.textContent = 'Upload or processing failed: ' + err;
    });
});
