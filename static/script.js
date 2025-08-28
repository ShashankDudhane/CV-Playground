const uploader = document.getElementById('uploader');
const inputImg = document.getElementById('inputImg');
const inputVideo = document.getElementById('inputVideo');
const outputCanvas = document.getElementById('output');
const runBtn = document.getElementById('runBtn');
const downloadBtn = document.getElementById('downloadBtn');
const webcamBtn = document.getElementById('webcamBtn');
const techSel = document.getElementById('technique');
const dynControls = document.getElementById('dynamicControls');
const cvStatus = document.getElementById('cvStatus');
const tabs = document.querySelectorAll('#sectionTabs .nav-link');
const explainPanel = document.getElementById('explainPanel');

let stream = null;
let useWebcam = false;
let currentSection = 'intensity';
let webcamLoopId = null;

const INTENSITY = [
  { id: 'negation', label: 'Image Negation' },
  { id: 'threshold', label: 'Thresholding' },
  { id: 'gray_slice', label: 'Gray-level Slicing' },
  { id: 'bit_plane', label: 'Bit-plane Slicing' },
];
const SMOOTHING = [
  { id: 'mean', label: 'Mean / Box Filter' },
  { id: 'gaussian', label: 'Gaussian Filter' },
  { id: 'median', label: 'Median Filter' },
];
const EDGES = [
  { id: 'sobel', label: 'Sobel' },
  { id: 'prewitt', label: 'Prewitt' },
  { id: 'roberts', label: 'Roberts' },
];
const CORNERS = [
  { id: 'shi', label: 'Shi–Tomasi Corners' },
  { id: 'harris', label: 'Harris Corners' },
];
const FEATURES = [
  { id: 'sift', label: 'SIFT (Scale-Invariant Feature Transform)' },
  { id: 'surf', label: 'SURF (Speeded-Up Robust Features)' },
];

function populateTechniques() {
  techSel.innerHTML = '';
  const list = currentSection === 'intensity' ? INTENSITY :
               currentSection === 'smoothing' ? SMOOTHING :
               currentSection === 'edge' ? EDGES :
               currentSection === 'corner' ? CORNERS :
               FEATURES;
  for (const t of list) {
    const opt = document.createElement('option');
    opt.value = t.id; opt.textContent = t.label;
    techSel.appendChild(opt);
  }
  buildDynamicControls();
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentSection = tab.dataset.section;
    populateTechniques();
  });
});

uploader.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  inputImg.onload = () => {
    URL.revokeObjectURL(url);
    matchCanvasToImage();
    runBtn.disabled = false;
  };
  inputImg.hidden = false;
  inputVideo.hidden = true;
  inputImg.src = url;
});

function matchCanvasToImage() {
  if (inputImg.naturalWidth) {
    outputCanvas.width = inputImg.naturalWidth;
    outputCanvas.height = inputImg.naturalHeight;
  }
}

function buildDynamicControls() {
  dynControls.innerHTML = '';
  const tech = techSel.value;

  const addRange = (id, label, min, max, step, value) => {
    const wrap = document.createElement('div');
    const lab = document.createElement('label'); lab.htmlFor = id;
    lab.textContent = `${label}: `;
    const rng = document.createElement('input');
    rng.type = 'range'; rng.id = id; rng.min = min; rng.max = max; rng.step = step; rng.value = value;
    const val = document.createElement('span'); val.textContent = value;
    rng.addEventListener('input', () => val.textContent = rng.value);
    wrap.appendChild(lab); wrap.appendChild(rng); wrap.appendChild(val);
    dynControls.appendChild(wrap);
  };
  const addSelect = (id, label, options) => {
    const wrap = document.createElement('div');
    const lab = document.createElement('label'); lab.htmlFor = id;
    lab.textContent = `${label}: `;
    const sel = document.createElement('select'); sel.id = id; sel.className = 'form-select form-select-sm';
    options.forEach(o => {
      const opt = document.createElement('option'); opt.value = o.value; opt.textContent = o.label;
      sel.appendChild(opt);
    });
    wrap.appendChild(lab); wrap.appendChild(sel);
    dynControls.appendChild(wrap);
  };
  const addNumber = (id, label, min, max, step, value) => {
    const wrap = document.createElement('div');
    const lab = document.createElement('label'); lab.htmlFor = id;
    lab.textContent = `${label}: `;
    const inp = document.createElement('input'); inp.type = 'number'; inp.id = id; inp.min = min; inp.max = max; inp.step = step; inp.value = value; inp.className = 'form-control form-control-sm'; inp.style.width = '6rem';
    wrap.appendChild(lab); wrap.appendChild(inp);
    dynControls.appendChild(wrap);
  };

  if (currentSection === 'intensity') {
    if (tech === 'threshold') {
      addSelect('th_type','Type',[{value:'binary',label:'Binary'},{value:'binary_inv',label:'Binary Inv'},{value:'trunc',label:'Truncate'},{value:'tozero',label:'To Zero'},{value:'tozero_inv',label:'To Zero Inv'}]);
      addRange('th_val','Threshold',0,255,1,128);
      addSelect('th_max','Max Value',[{value:'255',label:'255'},{value:'200',label:'200'},{value:'100',label:'100'}]);
    }
    if (tech === 'bit_plane') {
      addRange('bp_plane','Bit Plane',0,7,1,7);
    }
    if (tech === 'gray_slice') {
      addNumber('gs_low','Low',0,255,1,100);
      addNumber('gs_high','High',0,255,1,200);
      addSelect('gs_preserve','Preserve Outside',[{value:'yes',label:'Yes'},{value:'no',label:'No'}]);
    }
  }

  if (currentSection === 'smoothing') {
    if (tech === 'mean') addSelect('mean_k','Kernel',[{value:'3',label:'3x3'},{value:'5',label:'5x5'},{value:'7',label:'7x7'}]);
    if (tech === 'gaussian') addSelect('gaussK','Kernel',[{value:'3',label:'3x3'},{value:'5',label:'5x5'},{value:'7',label:'7x7'}]);
    if (tech === 'median') addSelect('medianK','Kernel',[{value:'3',label:'3'},{value:'5',label:'5'},{value:'7',label:'7'}]);
  }

  if (currentSection === 'edge') {
    if (tech === 'sobel') addSelect('sobel_k','Kernel',[{value:'3',label:'3'},{value:'5',label:'5'}]);
  }

  if (currentSection === 'corner') {
    if (tech === 'shi') {
      addRange('shi_n','Max Corners',50,1000,10,200);
      addRange('shi_q','Quality (×100)',1,100,1,10);
      addRange('shi_d','Min Distance',1,50,1,5);
    }
    if (tech === 'harris') {
      addNumber('h_block','Block Size',1,10,1,2);
      addNumber('h_ksize','Aperture (ksize)',1,7,2,3);
      addRange('h_k','k (×100)',1,200,1,4);
    }
  }

  if (currentSection === 'features') {
    addNumber('f_nfeatures','Max Features (ORB fallback)',100,5000,10,500);
    addNumber('f_octave','Octave Layers (SIFT/SURF)',1,10,1,3);
    addRange('f_contrast','Contrast Threshold (×1000)',1,500,1,50);
  }
}

techSel.addEventListener('change', buildDynamicControls);

// Build initial techniques list
populateTechniques();

// Helper: collect dynamic control values into a FormData-like object
function gatherParams(formData) {
  // read inputs inside dynControls
  const inputs = dynControls.querySelectorAll('input, select');
  inputs.forEach(inp => {
    if (inp.type === 'range' || inp.type === 'number' || inp.tagName.toLowerCase() === 'select' || inp.type === 'text') {
      formData.append(inp.id, inp.value);
    }
  });
}

// Display image blob into output canvas
function drawBlobToCanvas(blob) {
  const img = new Image();
  img.onload = () => {
    outputCanvas.width = img.naturalWidth;
    outputCanvas.height = img.naturalHeight;
    const ctx = outputCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
  };
  img.src = URL.createObjectURL(blob);
}

// Render explanation steps (array of {title, html})
function renderExplanation(json) {
  if (!json || !json.steps) {
    explainPanel.innerHTML = "<div class='text-danger'>No explanation returned.</div>";
    return;
  }
  let html = "";
  json.steps.forEach(s => {
    html += `<div class="mb-3"><h6 class="mb-1">${s.title}</h6><div>${s.html}</div></div>`;
  });
  explainPanel.innerHTML = html;
}

// Call explain endpoint (multipart form) and render
async function fetchAndRenderExplain(form) {
  try {
    const res = await fetch('/explain', { method: 'POST', body: form });
    if (!res.ok) {
      const txt = await res.text();
      explainPanel.innerHTML = `<div class="text-danger">Explain error: ${txt}</div>`;
      return;
    }
    const json = await res.json();
    renderExplanation(json);
  } catch (e) {
    explainPanel.innerHTML = `<div class="text-danger">Explain request failed: ${e.message}</div>`;
  }
}

// --- Single-image processing ---
runBtn.addEventListener('click', async () => {
  if (!inputImg.src && !useWebcam) return;
  const form = new FormData();
  form.append('section', currentSection);
  form.append('technique', techSel.value);
  gatherParams(form);

  if (useWebcam) {
    // If webcam mode is active, do single capture from video
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = inputVideo.videoWidth;
    tempCanvas.height = inputVideo.videoHeight;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(inputVideo, 0, 0);
    const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/png'));
    form.append('frame', blob, 'frame.png');
    const res = await fetch('/process_frame', { method: 'POST', body: form });
    const b = await res.blob();
    drawBlobToCanvas(b);
    downloadBtn.disabled = false;

    // fetch explanation for this frame
    const form2 = new FormData();
    form2.append('section', currentSection);
    form2.append('technique', techSel.value);
    gatherParams(form2);
    form2.append('frame', blob, 'frame.png');
    fetchAndRenderExplain(form2);
    return;
  }

  // file-based: need to fetch the file from uploader input
  const fileInput = uploader;
  const file = fileInput.files?.[0];
  if (!file) {
    alert('Please upload an image file first.');
    return;
  }
  form.append('image', file, file.name);

  const res = await fetch('/process', { method: 'POST', body: form });
  if (!res.ok) {
    const txt = await res.text();
    alert('Processing error: ' + txt);
    return;
  }
  const blob = await res.blob();
  drawBlobToCanvas(blob);
  downloadBtn.disabled = false;

  // fetch explanation
  const form2 = new FormData();
  form2.append('section', currentSection);
  form2.append('technique', techSel.value);
  gatherParams(form2);
  form2.append('image', file, file.name);
  fetchAndRenderExplain(form2);
});

// --- Download result ---
downloadBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'result.png';
  link.href = outputCanvas.toDataURL('image/png');
  link.click();
});

// --- Webcam handling: capture frames and stream to backend continuously ---
webcamBtn.addEventListener('click', async () => {
  if (useWebcam) { stopWebcam(); return; }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Webcam not supported by this browser.');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    inputVideo.srcObject = stream;
    inputVideo.hidden = false;
    inputImg.hidden = true;
    inputVideo.addEventListener('loadedmetadata', () => {
      inputVideo.width = inputVideo.videoWidth;
      inputVideo.height = inputVideo.videoHeight;
      useWebcam = true;
      webcamBtn.textContent = 'Stop Webcam';
      // start streaming frames to backend
      startWebcamLoop();
    }, { once: true });
  } catch (err) {
    alert('Could not access webcam: ' + err.message);
  }
});

function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }
  useWebcam = false;
  inputVideo.hidden = true;
  webcamBtn.textContent = '📷 Webcam';
  if (webcamLoopId) {
    cancelAnimationFrame(webcamLoopId);
    webcamLoopId = null;
  }
}

async function startWebcamLoop() {
  const loop = async () => {
    if (!useWebcam) return;
    if (inputVideo.videoWidth === 0 || inputVideo.videoHeight === 0) {
      webcamLoopId = requestAnimationFrame(loop);
      return;
    }
    // capture frame to canvas
    const temp = document.createElement('canvas');
    temp.width = inputVideo.videoWidth;
    temp.height = inputVideo.videoHeight;
    const ctx = temp.getContext('2d');
    ctx.drawImage(inputVideo, 0, 0);
    const blob = await new Promise(resolve => temp.toBlob(resolve, 'image/png'));
    // prepare form
    const form = new FormData();
    form.append('section', currentSection);
    form.append('technique', techSel.value);
    gatherParams(form);
    form.append('frame', blob, 'frame.png');
    try {
      const res = await fetch('/process_frame', { method: 'POST', body: form });
      if (res.ok) {
        const b = await res.blob();
        drawBlobToCanvas(b);
        // for streaming, we avoid calling explain every frame (performance).
        // If you want explanation for current frame, press ▶ Run once (it will do a single explain)
      } else {
        // ignore errors during streaming
        // console.warn('frame error', await res.text());
      }
    } catch (e) {
      console.warn('frame post failed', e);
    }
    webcamLoopId = requestAnimationFrame(loop);
  };
  webcamLoopId = requestAnimationFrame(loop);
}
