/**
 * FashionSearch — Frontend Script
 *
 * Handles:
 *  - Tab switching between search modes
 *  - Text / Image / Hybrid search via Fetch API
 *  - Result rendering with product cards
 *  - Relevance feedback (Rocchio)
 *  - Evaluation mode
 *  - Product detail modal
 *  - Drag-and-drop image upload
 */

// ============================================================
// Configuration
// ============================================================
const API_BASE = (window.API_BASE || "http://localhost:5000").replace(/\/$/, "");

// ============================================================
// DOM References
// ============================================================
const tabBtns        = document.querySelectorAll(".tab-btn");
const panels         = {
  text:     document.getElementById("panel-text"),
  image:    document.getElementById("panel-image"),
  hybrid:   document.getElementById("panel-hybrid"),
  evaluate: document.getElementById("panel-evaluate"),
};
const resultsSection  = document.getElementById("results-section");
const resultsGrid     = document.getElementById("results-grid");
const resultsTitle    = document.getElementById("results-title");
const spinner         = document.getElementById("spinner");
const errorToast      = document.getElementById("error-toast");
const feedbackControls = document.getElementById("feedback-controls");
const applyFeedbackBtn = document.getElementById("apply-feedback-btn");

// Modal
const productModal    = document.getElementById("product-modal");
const modalClose      = document.getElementById("modal-close");
const modalImage      = document.getElementById("modal-image");
const modalTitle      = document.getElementById("modal-title");
const modalBrand      = document.getElementById("modal-brand");
const modalSoldPrice  = document.getElementById("modal-sold-price");
const modalActualPrice = document.getElementById("modal-actual-price");
const modalUrl        = document.getElementById("modal-url");

// ============================================================
// State
// ============================================================
let currentMode      = "text";
let currentQuery     = "";
let lastResults      = [];
const feedbackState  = {};   // { [productId]: "relevant" | "not_relevant" }

// ============================================================
// Tab Switching
// ============================================================
tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.mode;
    if (mode === currentMode) return;

    // Update tabs
    tabBtns.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");

    // Update panels
    Object.entries(panels).forEach(([key, el]) => {
      el.classList.toggle("hidden", key !== mode);
    });

    currentMode = mode;
    hideResults();
    clearFeedback();
  });
});

// ============================================================
// Text Search
// ============================================================
document.getElementById("form-text").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query  = document.getElementById("text-query").value.trim();
  const topK   = parseInt(document.getElementById("text-top-k").value, 10) || 12;
  const expand = document.getElementById("text-expand").checked;
  const brand  = document.getElementById("text-brand").value.trim() || null;
  const minP   = parseFloat(document.getElementById("text-min-price").value) || null;
  const maxP   = parseFloat(document.getElementById("text-max-price").value) || null;

  if (!query) return;
  currentQuery = query;
  clearFeedback();

  const body = {
    query, top_k: topK, expand,
    ...(brand && { brand_filter: brand }),
    ...(minP != null && { min_price: minP }),
    ...(maxP != null && { max_price: maxP }),
  };

  const data = await apiRequest("POST", "/search/text", body, "json");
  if (data) {
    renderResults(data.results, `Text Results for "${query}" (${data.count})`);
    feedbackControls.classList.remove("hidden");
  }
});

// ============================================================
// Image Search
// ============================================================
const imageInput         = document.getElementById("image-input");
const imgSearchBtn       = document.getElementById("img-search-btn");
const previewImage       = document.getElementById("preview-image");
const uploadPlaceholder  = document.getElementById("upload-placeholder");
const uploadArea         = document.getElementById("upload-area");
const browseBtn          = document.getElementById("browse-btn");

browseBtn.addEventListener("click", () => imageInput.click());
uploadArea.addEventListener("click", (e) => {
  if (e.target !== browseBtn) imageInput.click();
});

imageInput.addEventListener("change", () => handleImageFile(imageInput.files[0]));

// Drag and drop
["dragover", "dragenter"].forEach((ev) =>
  uploadArea.addEventListener(ev, (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((ev) =>
  uploadArea.addEventListener(ev, (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    if (ev === "drop" && e.dataTransfer.files[0]) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  })
);

function handleImageFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImage.src = ev.target.result;
    previewImage.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");
    imgSearchBtn.disabled = false;
    imageInput._file = file;
  };
  reader.readAsDataURL(file);
}

document.getElementById("form-image").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file  = imageInput._file;
  if (!file) return showError("Please select an image first.");

  const topK  = parseInt(document.getElementById("img-top-k").value, 10) || 12;
  const brand = document.getElementById("img-brand").value.trim() || null;
  const minP  = parseFloat(document.getElementById("img-min-price").value) || null;
  const maxP  = parseFloat(document.getElementById("img-max-price").value) || null;

  const params = new URLSearchParams({ top_k: topK });
  if (brand) params.set("brand", brand);
  if (minP != null) params.set("min_price", minP);
  if (maxP != null) params.set("max_price", maxP);

  const formData = new FormData();
  formData.append("image", file);

  const data = await apiRequest(
    "POST",
    `/search/image?${params}`,
    formData,
    "form"
  );
  if (data) {
    renderResults(data.results, `Image Search Results (${data.count})`);
    feedbackControls.classList.add("hidden");
  }
});

// ============================================================
// Hybrid Search
// ============================================================
const hybridImageInput    = document.getElementById("hybrid-image-input");
const hybridPreview       = document.getElementById("hybrid-preview-image");
const hybridPlaceholder   = document.getElementById("hybrid-upload-placeholder");
const hybridUploadArea    = document.getElementById("hybrid-upload-area");
const hybridBrowseBtn     = document.getElementById("hybrid-browse-btn");
const hybridSearchBtn     = document.getElementById("hybrid-search-btn");
const hybridAlpha         = document.getElementById("hybrid-alpha");
const hybridAlphaVal      = document.getElementById("hybrid-alpha-val");

hybridBrowseBtn.addEventListener("click", () => hybridImageInput.click());
hybridUploadArea.addEventListener("click", (e) => {
  if (e.target !== hybridBrowseBtn) hybridImageInput.click();
});

hybridAlpha.addEventListener("input", () => {
  hybridAlphaVal.textContent = hybridAlpha.value;
});

hybridImageInput.addEventListener("change", () => {
  const file = hybridImageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    hybridPreview.src = ev.target.result;
    hybridPreview.classList.remove("hidden");
    hybridPlaceholder.classList.add("hidden");
    hybridSearchBtn.disabled = false;
    hybridImageInput._file = file;
  };
  reader.readAsDataURL(file);
});

document.getElementById("form-hybrid").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = document.getElementById("hybrid-query").value.trim();
  const file  = hybridImageInput._file;
  if (!query) return showError("Please enter a text query.");
  if (!file)  return showError("Please upload a reference image.");

  const topK  = parseInt(document.getElementById("hybrid-top-k").value, 10) || 12;
  const alpha = parseFloat(hybridAlpha.value);

  const params = new URLSearchParams({ top_k: topK });
  const formData = new FormData();
  formData.append("query", query);
  formData.append("image", file);
  formData.append("alpha", alpha);

  const data = await apiRequest(
    "POST",
    `/search/hybrid?${params}`,
    formData,
    "form"
  );
  if (data) {
    renderResults(data.results, `Hybrid Results for "${query}" (${data.count})`);
    feedbackControls.classList.add("hidden");
  }
});

// ============================================================
// Evaluate
// ============================================================
document.getElementById("form-evaluate").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query    = document.getElementById("eval-query").value.trim();
  const relevant = document.getElementById("eval-relevant").value
    .split(",").map((s) => s.trim()).filter(Boolean);
  const kValues  = document.getElementById("eval-k-values").value
    .split(",").map((s) => parseInt(s.trim(), 10)).filter(Boolean);

  if (!query) return showError("Query is required.");
  if (!relevant.length) return showError("At least one relevant ID is required.");

  const data = await apiRequest("POST", "/search/evaluate", {
    query,
    relevant_ids: relevant,
    k_values: kValues,
  }, "json");

  if (data) renderEvalResults(data.metrics);
});

// ============================================================
// Relevance Feedback
// ============================================================
applyFeedbackBtn.addEventListener("click", async () => {
  const positiveIds = Object.entries(feedbackState)
    .filter(([, v]) => v === "relevant").map(([k]) => k);
  const negativeIds = Object.entries(feedbackState)
    .filter(([, v]) => v === "not_relevant").map(([k]) => k);

  if (!positiveIds.length && !negativeIds.length) {
    return showError("Mark at least one result as relevant or not-relevant.");
  }

  const data = await apiRequest("POST", "/search/feedback", {
    query: currentQuery,
    positive_ids: positiveIds,
    negative_ids: negativeIds,
  }, "json");

  if (data) {
    clearFeedback();
    renderResults(data.results, `Refined Results (${data.count})`);
    feedbackControls.classList.remove("hidden");
  }
});

function clearFeedback() {
  Object.keys(feedbackState).forEach((k) => delete feedbackState[k]);
}

// ============================================================
// API Helper
// ============================================================
async function apiRequest(method, path, body, bodyType) {
  showSpinner();
  try {
    const opts = {
      method,
      headers: {},
    };

    if (bodyType === "json") {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    } else {
      opts.body = body; // FormData
    }

    const res = await fetch(`${API_BASE}${path}`, opts);
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || `Server error ${res.status}`);
      return null;
    }
    return data;
  } catch (err) {
    showError(`Network error: ${err.message}`);
    return null;
  } finally {
    hideSpinner();
  }
}

// ============================================================
// Rendering
// ============================================================
function renderResults(results, title) {
  lastResults = results;
  resultsTitle.textContent = title;
  resultsGrid.innerHTML = "";

  if (!results || results.length === 0) {
    resultsGrid.innerHTML =
      '<p style="color:var(--color-muted);padding:2rem 0">No results found. Try a different query.</p>';
    showResults();
    return;
  }

  results.forEach((product) => {
    const card = createProductCard(product);
    resultsGrid.appendChild(card);
  });

  showResults();
}

function createProductCard(product) {
  const card = document.createElement("div");
  card.className = "product-card";
  card.dataset.id = product.id;

  const pct = product.score != null ? Math.round(product.score * 100) : null;

  card.innerHTML = `
    <div class="card-image-wrap">
      <img
        src="${escHtml(product.image || product.img_url || "")}"
        alt="${escHtml(product.title)}"
        loading="lazy"
        onerror="this.src='https://via.placeholder.com/210x263?text=No+Image'"
      />
    </div>
    ${pct != null ? `<span class="card-score">${pct}%</span>` : ""}
    <div class="card-body">
      <div class="card-title">${escHtml(product.title)}</div>
      <div class="card-brand">${escHtml(product.brand)}</div>
      <div class="card-prices">
        <span class="price-sold">${escHtml(product.sold_price)}</span>
        ${product.actual_price ? `<span class="price-actual">${escHtml(product.actual_price)}</span>` : ""}
      </div>
    </div>
    <div class="card-feedback">
      <button class="feedback-btn relevant-btn" data-id="${escHtml(product.id)}" title="Mark relevant">👍 Relevant</button>
      <button class="feedback-btn not-relevant-btn" data-id="${escHtml(product.id)}" title="Mark not relevant">👎 Not Relevant</button>
    </div>
  `;

  // Click card body to open modal
  card.querySelector(".card-body").addEventListener("click", () => openModal(product));
  card.querySelector(".card-image-wrap").addEventListener("click", () => openModal(product));

  // Feedback buttons
  card.querySelector(".relevant-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleFeedback(product.id, "relevant", card);
  });
  card.querySelector(".not-relevant-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleFeedback(product.id, "not_relevant", card);
  });

  return card;
}

function toggleFeedback(id, type, card) {
  if (feedbackState[id] === type) {
    delete feedbackState[id];
    card.classList.remove("relevant", "not-relevant");
    card.querySelectorAll(".feedback-btn").forEach((b) => b.classList.remove("active"));
  } else {
    feedbackState[id] = type;
    card.classList.remove("relevant", "not-relevant");
    card.classList.add(type === "relevant" ? "relevant" : "not-relevant");
    card.querySelectorAll(".feedback-btn").forEach((b) => b.classList.remove("active"));
    const btnClass = type === "relevant" ? ".relevant-btn" : ".not-relevant-btn";
    card.querySelector(btnClass).classList.add("active");
  }
}

function renderEvalResults(metrics) {
  const container = document.getElementById("eval-results");
  container.innerHTML = "<h3>Evaluation Metrics</h3>";
  const grid = document.createElement("div");
  grid.className = "metrics-grid";

  Object.entries(metrics).forEach(([name, value]) => {
    const mc = document.createElement("div");
    mc.className = "metric-card";
    mc.innerHTML = `
      <div class="metric-name">${escHtml(name)}</div>
      <div class="metric-value">${(value * 100).toFixed(1)}%</div>
    `;
    grid.appendChild(mc);
  });

  container.appendChild(grid);
  container.classList.remove("hidden");
}

// ============================================================
// Modal
// ============================================================
function openModal(product) {
  modalImage.src    = product.image || product.img_url || "";
  modalImage.alt    = product.title;
  modalTitle.textContent      = product.title;
  modalBrand.textContent      = product.brand ? `Brand: ${product.brand}` : "";
  modalSoldPrice.textContent  = product.sold_price || "";
  modalActualPrice.textContent = product.actual_price || "";
  modalUrl.href = product.url || "#";
  modalUrl.style.display = product.url ? "" : "none";

  productModal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  productModal.classList.add("hidden");
  document.body.style.overflow = "";
}

modalClose.addEventListener("click", closeModal);
productModal.addEventListener("click", (e) => {
  if (e.target === productModal) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

// ============================================================
// UI Helpers
// ============================================================
function showSpinner()  { spinner.classList.remove("hidden"); hideResults(); }
function hideSpinner()  { spinner.classList.add("hidden"); }
function showResults()  { resultsSection.classList.remove("hidden"); }
function hideResults()  {
  resultsSection.classList.add("hidden");
  document.getElementById("eval-results").classList.add("hidden");
}

let _errorTimer;
function showError(msg) {
  errorToast.textContent = msg;
  errorToast.classList.remove("hidden");
  clearTimeout(_errorTimer);
  _errorTimer = setTimeout(() => errorToast.classList.add("hidden"), 4000);
}

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
