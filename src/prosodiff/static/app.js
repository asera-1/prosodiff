"use strict";

const form = document.getElementById("comparison-form");
const fileInput = document.getElementById("wav-input");
const fileStatus = document.getElementById("file-status");
const labelsFieldset = document.getElementById("take-labels");
const takeList = document.getElementById("take-list");
const protocolAck = document.getElementById("protocol-ack");
const generateButton = document.getElementById("generate-button");
const loadingState = document.getElementById("loading-state");
const previewStage = document.getElementById("preview-stage");

if (form && fileInput) {
  const defaultLabel = (filename, index) => {
    const withoutExtension = filename.replace(/\.[^.]+$/, "");
    const words = withoutExtension.replace(/[_-]+/g, " ").trim();
    return words
      ? words.replace(/\b\w/g, (character) => character.toUpperCase()).slice(0, 64)
      : `Take ${index}`;
  };

  const readableSize = (bytes) => {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const updateReadyState = () => {
    const countIsValid = fileInput.files.length >= 2 && fileInput.files.length <= 4;
    generateButton.disabled = !(countIsValid && protocolAck.checked);
  };

  const renderFiles = () => {
    const files = Array.from(fileInput.files);
    takeList.replaceChildren();
    labelsFieldset.hidden = files.length === 0;

    const validCount = files.length >= 2 && files.length <= 4;
    fileInput.setCustomValidity(
      files.length < 2
        ? "Select at least two WAV recordings."
        : files.length > 4
          ? "Prosodiff accepts at most four recordings."
          : ""
    );
    fileStatus.textContent = files.length === 0
      ? "No recordings selected"
      : validCount
        ? `${files.length} recordings ready`
        : `${files.length} recordings selected · choose between 2 and 4`;
    fileStatus.dataset.valid = String(validCount);

    files.forEach((file, index) => {
      const row = document.createElement("div");
      row.className = `take-row take-${index + 1}`;

      const marker = document.createElement("span");
      marker.className = "take-symbol";
      marker.setAttribute("aria-hidden", "true");

      const copy = document.createElement("div");
      copy.className = "take-file-copy";
      const filename = document.createElement("strong");
      filename.textContent = file.name;
      const size = document.createElement("small");
      size.textContent = `T${index + 1} · ${readableSize(file.size)}${index === 0 ? " · delta reference" : ""}`;
      copy.append(filename, size);

      const label = document.createElement("label");
      label.textContent = "Label";
      const input = document.createElement("input");
      input.type = "text";
      input.name = "labels";
      input.maxLength = 64;
      input.required = true;
      input.value = defaultLabel(file.name, index + 1);
      input.setAttribute("aria-label", `Label for take ${index + 1}`);
      label.append(input);

      row.append(marker, copy, label);
      takeList.append(row);
    });
    updateReadyState();
  };

  fileInput.addEventListener("change", renderFiles);
  protocolAck.addEventListener("change", updateReadyState);

  form.addEventListener("submit", (event) => {
    const labels = Array.from(form.querySelectorAll('input[name="labels"]'));
    const uniqueLabels = new Set(labels.map((input) => input.value.trim().toLocaleLowerCase()));
    if (uniqueLabels.size !== labels.length) {
      event.preventDefault();
      labels.forEach((input) => input.setCustomValidity("Take labels must be unique."));
      labels.find((input) => input.value.trim())?.reportValidity();
      return;
    }
    labels.forEach((input) => input.setCustomValidity(""));
    if (!form.checkValidity()) {
      event.preventDefault();
      form.reportValidity();
      return;
    }
    form.setAttribute("aria-busy", "true");
    generateButton.disabled = true;
    generateButton.firstElementChild.textContent = "Generating…";
    previewStage.classList.add("is-loading");
    loadingState.hidden = false;
  });

  renderFiles();
}
