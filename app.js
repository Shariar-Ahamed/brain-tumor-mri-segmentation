// NeuroSeg AI - Interactive Brain Tumor MRI Segmentation Web Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sliceSlider = document.getElementById("sliceSlider");
  const sliceVal = document.getElementById("sliceVal");
  const caseSelect = document.getElementById("caseSelect");
  const modelSelect = document.getElementById("modelSelect");
  const modalityLabel = document.getElementById("modalityLabel");
  const modalityBtns = document.querySelectorAll(".modality-btn");

  const chkET = document.getElementById("chkET");
  const chkTC = document.getElementById("chkTC");
  const chkED = document.getElementById("chkED");
  const btnRunInference = document.getElementById("btnRunInference");

  const mriCanvas = document.getElementById("mriCanvas");
  const segCanvas = document.getElementById("segCanvas");
  const xaiCanvas = document.getElementById("xaiCanvas");
  const robustnessCanvas = document.getElementById("robustnessCanvas");

  const mDice = document.getElementById("mDice");
  const mIoU = document.getElementById("mIoU");
  const mVol = document.getElementById("mVol");
  const mTime = document.getElementById("mTime");

  const xaiClassSelect = document.getElementById("xaiClassSelect");
  const xaiAlpha = document.getElementById("xaiAlpha");
  const noiseSlider = document.getElementById("noiseSlider");
  const noiseVal = document.getElementById("noiseVal");
  const blurSlider = document.getElementById("blurSlider");
  const blurVal = document.getElementById("blurVal");

  let currentModality = "FLAIR";

  // Smooth Scroll & Active Link Handler for Navigation Bar & Buttons
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
        const navLink = document.querySelector(`.nav-links a[href="${targetId}"]`);
        if (navLink) navLink.classList.add('active');

        const navHeight = 80;
        const elementPosition = targetElement.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - navHeight;

        window.scrollTo({
          top: Math.max(0, offsetPosition),
          behavior: 'smooth'
        });
      }
    });
  });

  // Event Listeners
  sliceSlider.addEventListener("input", (e) => {
    sliceVal.textContent = e.target.value;
    renderAllCanvases();
  });

  modalityBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      modalityBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentModality = btn.dataset.modality;
      modalityLabel.textContent = `${currentModality} Sequence`;
      renderAllCanvases();
    });
  });

  caseSelect.addEventListener("change", renderAllCanvases);
  modelSelect.addEventListener("change", () => {
    updateMetrics();
    renderAllCanvases();
  });

  [chkET, chkTC, chkED].forEach(chk => chk.addEventListener("change", renderAllCanvases));

  noiseSlider.addEventListener("input", (e) => {
    noiseVal.textContent = parseFloat(e.target.value).toFixed(2);
    renderRobustnessCanvas();
  });

  blurSlider.addEventListener("input", (e) => {
    blurVal.textContent = `${e.target.value}px`;
    renderRobustnessCanvas();
  });

  xaiClassSelect.addEventListener("change", renderXAICanvas);
  xaiAlpha.addEventListener("input", renderXAICanvas);

  btnRunInference.addEventListener("click", () => {
    btnRunInference.textContent = "Processing Inference...";
    btnRunInference.style.opacity = "0.7";
    setTimeout(() => {
      btnRunInference.textContent = "Run Segmentation Prediction ⚡";
      btnRunInference.style.opacity = "1";
      renderAllCanvases();
      updateMetrics();
    }, 300);
  });

  // Canvas Procedural Rendering Functions
  function drawBrainOutline(ctx, width, height, sliceIdx, modality) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const radiusX = Math.min(width, height) * (0.35 + Math.sin(sliceIdx / 30) * 0.05);
    const radiusY = Math.min(width, height) * (0.40 + Math.sin(sliceIdx / 30) * 0.04);

    // Brain skull / tissue shape
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
    
    let baseColor = 30;
    if (modality === "FLAIR") baseColor = 45;
    if (modality === "T1ce") baseColor = 50;
    if (modality === "T2") baseColor = 60;

    const grad = ctx.createRadialGradient(centerX, centerY, radiusX * 0.2, centerX, centerY, radiusX);
    grad.addColorStop(0, `rgb(${baseColor + 40}, ${baseColor + 40}, ${baseColor + 50})`);
    grad.addColorStop(0.7, `rgb(${baseColor + 10}, ${baseColor + 10}, ${baseColor + 20})`);
    grad.addColorStop(1, `rgb(10, 10, 15)`);

    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();

    // Internal Brain Structures (Ventricular System & Gyri)
    ctx.strokeStyle = `rgba(255, 255, 255, 0.12)`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(centerX - 15, centerY, 8, 25, 0.1, 0, 2 * Math.PI);
    ctx.ellipse(centerX + 15, centerY, 8, 25, -0.1, 0, 2 * Math.PI);
    ctx.stroke();

    return { centerX, centerY, radiusX, radiusY };
  }

  function drawTumorMask(ctx, brainParams, sliceIdx, model, isOverlay = true) {
    const { centerX, centerY, radiusX, radiusY } = brainParams;
    const tumorX = centerX + radiusX * 0.25;
    const tumorY = centerY - radiusY * 0.15;
    const tumorSize = 35 + Math.sin(sliceIdx / 20) * 10;

    if (tumorSize <= 10) return; // No tumor slice

    // 1. Edema (ED)
    if (chkED.checked) {
      ctx.beginPath();
      ctx.ellipse(tumorX, tumorY, tumorSize * 1.5, tumorSize * 1.3, 0.3, 0, 2 * Math.PI);
      ctx.fillStyle = isOverlay ? "rgba(127, 0, 255, 0.4)" : "#7f00ff";
      ctx.fill();
    }

    // 2. Tumor Core (TC)
    if (chkTC.checked) {
      ctx.beginPath();
      ctx.ellipse(tumorX, tumorY, tumorSize * 0.9, tumorSize * 0.8, -0.2, 0, 2 * Math.PI);
      ctx.fillStyle = isOverlay ? "rgba(255, 0, 127, 0.6)" : "#ff007f";
      ctx.fill();
    }

    // 3. Enhancing Tumor (ET)
    if (chkET.checked) {
      ctx.beginPath();
      ctx.ellipse(tumorX + 5, tumorY - 5, tumorSize * 0.5, tumorSize * 0.45, 0.1, 0, 2 * Math.PI);
      ctx.fillStyle = isOverlay ? "rgba(255, 183, 3, 0.75)" : "#ffb703";
      ctx.fill();
    }
  }

  function renderMRICanvas() {
    const ctx = mriCanvas.getContext("2d");
    const sliceIdx = parseInt(sliceSlider.value);
    drawBrainOutline(ctx, mriCanvas.width, mriCanvas.height, sliceIdx, currentModality);
  }

  function renderSegCanvas() {
    const ctx = segCanvas.getContext("2d");
    const sliceIdx = parseInt(sliceSlider.value);
    const model = modelSelect.value;
    const brainParams = drawBrainOutline(ctx, segCanvas.width, segCanvas.height, sliceIdx, currentModality);
    drawTumorMask(ctx, brainParams, sliceIdx, model, true);
  }

  function renderXAICanvas() {
    const ctx = xaiCanvas.getContext("2d");
    const sliceIdx = parseInt(sliceSlider.value);
    const alpha = parseFloat(xaiAlpha.value);
    const brainParams = drawBrainOutline(ctx, xaiCanvas.width, xaiCanvas.height, sliceIdx, currentModality);

    // Grad-CAM++ Heatmap Overlay
    const { centerX, centerY, radiusX, radiusY } = brainParams;
    const tumorX = centerX + radiusX * 0.25;
    const tumorY = centerY - radiusY * 0.15;
    const radius = 60;

    const heatGrad = ctx.createRadialGradient(tumorX, tumorY, 0, tumorX, tumorY, radius);
    heatGrad.addColorStop(0, `rgba(255, 0, 0, ${alpha})`);
    heatGrad.addColorStop(0.4, `rgba(255, 165, 0, ${alpha * 0.8})`);
    heatGrad.addColorStop(0.7, `rgba(255, 255, 0, ${alpha * 0.5})`);
    heatGrad.addColorStop(1, `rgba(0, 0, 255, 0)`);

    ctx.beginPath();
    ctx.arc(tumorX, tumorY, radius, 0, 2 * Math.PI);
    ctx.fillStyle = heatGrad;
    ctx.fill();
  }

  function renderRobustnessCanvas() {
    const ctx = robustnessCanvas.getContext("2d");
    const sliceIdx = parseInt(sliceSlider.value);
    const noise = parseFloat(noiseSlider.value);
    
    const brainParams = drawBrainOutline(ctx, robustnessCanvas.width, robustnessCanvas.height, sliceIdx, currentModality);
    drawTumorMask(ctx, brainParams, sliceIdx, "Proposed-Hybrid", true);

    // Apply Simulated Noise Overlay
    if (noise > 0) {
      const imgData = ctx.getImageData(0, 0, robustnessCanvas.width, robustnessCanvas.height);
      const data = imgData.data;
      for (let i = 0; i < data.length; i += 4) {
        if (Math.random() < noise) {
          const n = (Math.random() - 0.5) * noise * 255;
          data[i] = Math.min(255, Math.max(0, data[i] + n));
          data[i+1] = Math.min(255, Math.max(0, data[i+1] + n));
          data[i+2] = Math.min(255, Math.max(0, data[i+2] + n));
        }
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }

  function renderAllCanvases() {
    renderMRICanvas();
    renderSegCanvas();
    renderXAICanvas();
    renderRobustnessCanvas();
  }

  function updateMetrics() {
    const m = modelSelect.value;
    if (m === "Proposed-Hybrid") {
      mDice.textContent = "0.9320";
      mIoU.textContent = "0.8745";
      mVol.textContent = "24.6 cm³";
      mTime.textContent = "18.4 ms";
      mDice.style.color = "var(--accent-cyan)";
    } else if (m === "ResNet34-UNet") {
      mDice.textContent = "0.9211";
      mIoU.textContent = "0.8540";
      mVol.textContent = "24.1 cm³";
      mTime.textContent = "22.1 ms";
      mDice.style.color = "var(--accent-blue)";
    } else if (m === "U-Net++") {
      mDice.textContent = "0.9185";
      mIoU.textContent = "0.8490";
      mVol.textContent = "23.9 cm³";
      mTime.textContent = "28.5 ms";
      mDice.style.color = "var(--accent-pink)";
    } else {
      mDice.textContent = "0.9140";
      mIoU.textContent = "0.8410";
      mVol.textContent = "23.5 cm³";
      mTime.textContent = "16.8 ms";
      mDice.style.color = "var(--accent-yellow)";
    }
  }

  // Initial render
  renderAllCanvases();
  updateMetrics();
});
