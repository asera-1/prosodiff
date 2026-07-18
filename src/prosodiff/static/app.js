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
const liveRecorder = document.getElementById("live-recorder");
const recorderTitle = document.getElementById("recorder-title");
const recordButton = document.getElementById("record-button");
const stopButton = document.getElementById("stop-button");
const recordingTime = document.getElementById("recording-time");
const recordingStatus = document.getElementById("recording-status");
const recordingError = document.getElementById("recording-error");
const levelValue = document.getElementById("level-value");
const recordedTakes = document.getElementById("recorded-takes");
const recordedList = document.getElementById("recorded-list");
const recordingCount = document.getElementById("recording-count");
const uploadOption = document.getElementById("upload-option");
const captureMetadataInput = document.getElementById("capture-metadata");

if (form && fileInput) {
  const MINIMUM_RECORDING_SECONDS = 0.35;
  const MAXIMUM_RECORDING_SECONDS = 30;
  const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;
  const supportsLiveRecording = Boolean(
    navigator.mediaDevices?.getUserMedia
      && AudioContextConstructor
      && window.DataTransfer
      && window.File
  );

  let recordings = [];
  let sourceMode = "record";
  let activeCapture = null;
  let timerId = null;

  const defaultLabel = (filename, index) => {
    const withoutExtension = filename.replace(/\.[^.]+$/, "");
    const words = withoutExtension.replace(/[_-]+/g, " ").trim();
    return words
      ? words.replace(/\b\w/g, (character) => character.toUpperCase()).slice(0, 64)
      : `Take ${index}`;
  };

  const recordedLabel = (index) => index === 0 ? "Reference" : `Delivery ${index + 1}`;

  const readableSize = (bytes) => {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds - minutes * 60;
    return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
  };

  const dbfs = (amplitude) => 20 * Math.log10(Math.max(amplitude, 1e-12));

  const quantile = (values, proportion) => {
    if (!values.length) return -120;
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * proportion;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const fraction = position - lower;
    return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
  };

  const flattenChunks = (chunks, maximumSamples) => {
    const available = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const length = Math.min(available, maximumSamples);
    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      if (offset >= length) break;
      const remaining = length - offset;
      samples.set(chunk.subarray(0, remaining), offset);
      offset += Math.min(chunk.length, remaining);
    }
    return samples;
  };

  const inspectCapture = (samples, sampleRate) => {
    let sum = 0;
    let peak = 0;
    let clippedSamples = 0;
    let clippedRun = 0;
    let hasClippedRun = false;
    for (const sample of samples) {
      sum += sample;
      const absolute = Math.abs(sample);
      peak = Math.max(peak, absolute);
      if (absolute >= 0.999) {
        clippedSamples += 1;
        clippedRun += 1;
        hasClippedRun ||= clippedRun >= 3;
      } else {
        clippedRun = 0;
      }
    }
    const mean = samples.length ? sum / samples.length : 0;
    const frameLength = Math.max(1, Math.round(sampleRate * 1024 / 22050));
    const hopLength = Math.max(1, Math.round(sampleRate * 256 / 22050));
    const frameLevels = [];
    for (let start = 0; start + frameLength <= samples.length; start += hopLength) {
      let squareSum = 0;
      for (let index = start; index < start + frameLength; index += 1) {
        const centered = samples[index] - mean;
        squareSum += centered * centered;
      }
      frameLevels.push(Math.max(-120, dbfs(Math.sqrt(squareSum / frameLength))));
    }
    const q10 = quantile(frameLevels, 0.10);
    const q90 = quantile(frameLevels, 0.90);
    const threshold = Math.min(q90 - 12, Math.max(q10 + 6, q90 - 35));
    const activeLevels = frameLevels.filter((value) => value >= threshold);
    return {
      activeRmsDbfs: quantile(activeLevels, 0.50),
      peakDbfs: dbfs(peak),
      peak,
      clippedFraction: samples.length ? clippedSamples / samples.length : 0,
      hasClippedRun,
    };
  };

  const trackValue = (settings, key) => (
    Object.prototype.hasOwnProperty.call(settings, key) ? settings[key] : null
  );

  const updateDerivedQa = () => {
    const reference = recordings[0];
    recordings.forEach((recording, index) => {
      const codes = [];
      const messages = [];
      if (recording.qa.activeRmsDbfs < -35) {
        codes.push("LOW_LEVEL");
        messages.push(`level ${recording.qa.activeRmsDbfs.toFixed(1)} dBFS is below the -35 dBFS review floor`);
      }
      if (recording.qa.clippedFraction >= 0.001 || recording.qa.hasClippedRun) {
        codes.push("CLIPPING");
        messages.push(`${(recording.qa.clippedFraction * 100).toFixed(2)}% of samples approach full scale`);
      }
      if (index > 0 && reference) {
        const difference = recording.qa.activeRmsDbfs - reference.qa.activeRmsDbfs;
        if (Math.abs(difference) > 6) {
          codes.push("LEVEL_MISMATCH");
          messages.push(`${Math.abs(difference).toFixed(1)} dB ${difference < 0 ? "quieter" : "louder"} than Reference`);
        }
      }
      const referenceDevice = reference?.deviceId || null;
      recording.sameDeviceAsReference = index === 0
        ? (recording.deviceId ? true : null)
        : (referenceDevice && recording.deviceId
          ? recording.deviceId === referenceDevice
          : null);
      if (recording.sameDeviceAsReference === false) {
        codes.push("DEVICE_CHANGED");
        messages.push("input device differs from Reference");
      }
      recording.qa.codes = codes;
      recording.qa.messages = messages;
      recording.qa.redoRecommended = codes.length > 0;
    });
  };

  const syncCaptureMetadata = () => {
    if (!captureMetadataInput) return;
    if (sourceMode !== "record" || recordings.length === 0) {
      captureMetadataInput.value = "";
      return;
    }
    captureMetadataInput.value = JSON.stringify({
      schema_version: "0.1.0",
      takes: recordings.map((recording) => ({
        source: "live",
        wav: {
          sample_rate_hz: recording.sampleRate,
          channels: 1,
          duration_s: recording.duration,
        },
        track: {
          sample_rate_hz: recording.trackSettings.sampleRate,
          channel_count: recording.trackSettings.channelCount,
          sample_size_bits: recording.trackSettings.sampleSize,
          latency_s: recording.trackSettings.latency,
          echo_cancellation: recording.trackSettings.echoCancellation,
          noise_suppression: recording.trackSettings.noiseSuppression,
          auto_gain_control: recording.trackSettings.autoGainControl,
          same_device_as_reference: recording.sameDeviceAsReference,
        },
        constraints_fallback: recording.constraintsFallback,
        client_qa: {
          active_rms_estimate_dbfs: recording.qa.activeRmsDbfs,
          peak_dbfs: recording.qa.peakDbfs,
          clipped_sample_fraction: recording.qa.clippedFraction,
          codes: recording.qa.codes,
        },
      })),
    });
  };

  const showRecordingError = (message = "") => {
    recordingError.textContent = message;
    recordingError.hidden = !message;
  };

  const updateReadyState = () => {
    const count = fileInput.files.length;
    const countIsValid = count >= 2 && count <= 4;
    generateButton.disabled = !(countIsValid && protocolAck.checked) || Boolean(activeCapture);
    recordButton.disabled = Boolean(activeCapture) || recordings.length >= 4 || !supportsLiveRecording;
    if (!activeCapture) {
      recordButton.lastElementChild.textContent = recordings.length >= 4
        ? "Four takes recorded"
        : `Record Take ${recordings.length + 1}`;
    }
  };

  const setFileValidity = () => {
    const count = fileInput.files.length;
    fileInput.setCustomValidity(
      count < 2
        ? "Record or select at least two WAV takes."
        : count > 4
          ? "Prosodiff accepts at most four takes."
          : ""
    );
  };

  const encodeWav = (chunks, sampleRate, maximumSamples = Infinity) => {
    const availableSamples = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const sampleCount = Math.min(availableSamples, maximumSamples);
    const buffer = new ArrayBuffer(44 + sampleCount * 2);
    const view = new DataView(buffer);

    const writeText = (offset, text) => {
      for (let index = 0; index < text.length; index += 1) {
        view.setUint8(offset + index, text.charCodeAt(index));
      }
    };

    writeText(0, "RIFF");
    view.setUint32(4, 36 + sampleCount * 2, true);
    writeText(8, "WAVE");
    writeText(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeText(36, "data");
    view.setUint32(40, sampleCount * 2, true);

    let offset = 44;
    let writtenSamples = 0;
    for (const chunk of chunks) {
      for (const sample of chunk) {
        if (writtenSamples >= sampleCount) break;
        const clamped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clamped < 0 ? clamped * 32768 : clamped * 32767, true);
        offset += 2;
        writtenSamples += 1;
      }
      if (writtenSamples >= sampleCount) break;
    }
    return new Blob([view], { type: "audio/wav" });
  };

  const syncRecordedFiles = () => {
    updateDerivedQa();
    const transfer = new DataTransfer();
    recordings.forEach((recording, index) => {
      transfer.items.add(new File(
        [recording.blob],
        `take_${index + 1}_recorded.wav`,
        { type: "audio/wav", lastModified: Date.now() }
      ));
    });
    fileInput.files = transfer.files;
    syncCaptureMetadata();
    setFileValidity();
  };

  const clearRecordedTakes = () => {
    recordings.forEach((recording) => URL.revokeObjectURL(recording.url));
    recordings = [];
    syncCaptureMetadata();
    recordedList.replaceChildren();
    recordedTakes.hidden = true;
    recordingCount.textContent = "";
  };

  const renderRecordedTakes = () => {
    updateDerivedQa();
    syncCaptureMetadata();
    recordedList.replaceChildren();
    recordedTakes.hidden = recordings.length === 0;

    recordings.forEach((recording, index) => {
      const article = document.createElement("article");
      article.className = `recorded-take take-${index + 1}`;

      const header = document.createElement("header");
      const marker = document.createElement("span");
      marker.className = "take-symbol";
      marker.setAttribute("aria-hidden", "true");
      const title = document.createElement("strong");
      title.textContent = `T${index + 1} · ${formatTime(recording.duration)}`;
      const reference = document.createElement("small");
      reference.textContent = index === 0 ? "delta reference" : "recorded live";
      header.append(marker, title, reference);

      const qa = document.createElement("div");
      qa.className = `take-qa${recording.qa.redoRecommended ? " is-review" : ""}`;
      const qaStatus = document.createElement("strong");
      qaStatus.textContent = recording.qa.redoRecommended
        ? "Redo recommended"
        : "Level check passed";
      const qaValues = document.createElement("span");
      qaValues.textContent = `speech RMS estimate ${recording.qa.activeRmsDbfs.toFixed(1)} dBFS · peak ${recording.qa.peakDbfs.toFixed(1)} dBFS`;
      qa.append(qaStatus, qaValues);
      if (recording.qa.messages.length) {
        const qaReasons = document.createElement("small");
        qaReasons.textContent = recording.qa.messages.join(" · ");
        qa.append(qaReasons);
      }

      const processingNames = [
        ["echo cancellation", recording.trackSettings.echoCancellation],
        ["noise suppression", recording.trackSettings.noiseSuppression],
        ["auto gain", recording.trackSettings.autoGainControl],
      ];
      const activeProcessing = processingNames
        .filter(([, value]) => value === true)
        .map(([name]) => name);
      const unknownProcessing = processingNames.some(([, value]) => value === null);
      let captureNoticeText = "";
      if (activeProcessing.length) {
        captureNoticeText = `Browser processing active: ${activeProcessing.join(", ")}.`;
      } else if (unknownProcessing) {
        captureNoticeText = "The browser did not report every processing setting.";
      } else if (recording.constraintsFallback) {
        captureNoticeText = "Capture started after retrying without exact constraints.";
      }
      const captureNotice = document.createElement("small");
      captureNotice.className = "capture-note";
      captureNotice.textContent = captureNoticeText;
      captureNotice.hidden = !captureNoticeText;

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = recording.url;
      audio.setAttribute("aria-label", `Playback for Take ${index + 1}`);

      const label = document.createElement("label");
      label.textContent = "Label";
      const input = document.createElement("input");
      input.type = "text";
      input.name = "labels";
      input.maxLength = 64;
      input.required = true;
      input.value = recording.label;
      input.setAttribute("aria-label", `Label for Take ${index + 1}`);
      input.addEventListener("input", () => {
        recordings[index].label = input.value;
      });
      label.append(input);

      const actions = document.createElement("div");
      actions.className = "recorded-actions";
      const redo = document.createElement("button");
      redo.type = "button";
      redo.textContent = "Redo";
      redo.addEventListener("click", () => beginRecording(index));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        URL.revokeObjectURL(recordings[index].url);
        recordings.splice(index, 1);
        recordings.forEach((item, itemIndex) => {
          if (!item.label || /^Delivery \d+$/.test(item.label)) {
            item.label = recordedLabel(itemIndex);
          }
        });
        syncRecordedFiles();
        renderRecordedTakes();
        recordButton.focus({ preventScroll: true });
      });
      actions.append(redo, remove);

      article.append(header, qa, captureNotice, audio, label, actions);
      recordedList.append(article);
    });

    recordingCount.textContent = recordings.length < 2
      ? "Record at least one more take."
      : `${recordings.length} takes ready · add another or generate the comparison.`;
    recordingStatus.textContent = recordings.length === 0
      ? "Start with a neutral or reference delivery."
      : recordings.length < 2
        ? "Now record the same sentence with a different delivery."
        : "Your matched takes are ready.";
    recorderTitle.textContent = "Microphone ready";
    updateReadyState();
  };

  const renderUploadedFiles = () => {
    const files = Array.from(fileInput.files);
    takeList.replaceChildren();
    labelsFieldset.hidden = files.length === 0;
    setFileValidity();

    const validCount = files.length >= 2 && files.length <= 4;
    fileStatus.textContent = files.length === 0
      ? "No recordings selected"
      : validCount
        ? `${files.length} existing recordings ready`
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
      const referenceNote = index === 0 ? " · delta reference" : "";
      size.textContent = `T${index + 1} · ${readableSize(file.size)}${referenceNote}`;
      copy.append(filename, size);
      const label = document.createElement("label");
      label.textContent = "Label";
      const input = document.createElement("input");
      input.type = "text";
      input.name = "labels";
      input.maxLength = 64;
      input.required = true;
      input.value = defaultLabel(file.name, index + 1);
      input.setAttribute("aria-label", `Label for Take ${index + 1}`);
      label.append(input);
      row.append(marker, copy, label);
      takeList.append(row);
    });
    updateReadyState();
  };

  const resetCaptureUi = () => {
    clearInterval(timerId);
    timerId = null;
    liveRecorder.classList.remove("is-recording");
    recordButton.hidden = false;
    stopButton.hidden = true;
    recordingTime.textContent = "00:00.0";
    recordingTime.setAttribute("datetime", "PT0S");
    levelValue.style.transform = "scaleX(0)";
    updateReadyState();
    recordButton.focus({ preventScroll: true });
  };

  const stopRecording = async () => {
    if (!activeCapture) return;
    const capture = activeCapture;
    activeCapture = null;
    capture.processor.onaudioprocess = null;
    for (const node of [capture.source, capture.processor, capture.silentGain]) {
      try {
        node.disconnect();
      } catch {
        // Continue until the microphone track and context are closed.
      }
    }
    capture.stream.getTracks().forEach((track) => track.stop());
    try {
      await capture.context.close();
    } catch {
      // The microphone track is already stopped; continue saving captured PCM.
    }
    resetCaptureUi();

    const maximumSamples = Math.floor(capture.sampleRate * MAXIMUM_RECORDING_SECONDS);
    const samples = flattenChunks(capture.chunks, maximumSamples);
    const duration = samples.length / capture.sampleRate;
    if (duration < MINIMUM_RECORDING_SECONDS) {
      showRecordingError("That take was too short. Record for at least 0.35 seconds.");
      recorderTitle.textContent = "Microphone ready";
      updateReadyState();
      return;
    }
    const qa = inspectCapture(samples, capture.sampleRate);
    if (qa.peak < 1e-6) {
      showRecordingError("No microphone signal was captured. Check the input and record again.");
      recorderTitle.textContent = "Microphone ready";
      updateReadyState();
      return;
    }

    const blob = encodeWav([samples], capture.sampleRate, samples.length);
    const previous = capture.replaceIndex === null ? null : recordings[capture.replaceIndex];
    const recording = {
      blob,
      duration,
      label: previous?.label || recordedLabel(capture.replaceIndex ?? recordings.length),
      url: URL.createObjectURL(blob),
      sampleRate: capture.sampleRate,
      trackSettings: capture.trackSettings,
      constraintsFallback: capture.constraintsFallback,
      deviceId: capture.deviceId,
      sameDeviceAsReference: null,
      qa: {
        ...qa,
        codes: [],
        messages: [],
        redoRecommended: false,
      },
    };
    if (capture.replaceIndex === null) {
      recordings.push(recording);
    } else {
      URL.revokeObjectURL(previous.url);
      recordings[capture.replaceIndex] = recording;
    }
    sourceMode = "record";
    syncRecordedFiles();
    renderRecordedTakes();
  };

  const updateCaptureClock = () => {
    if (!activeCapture) return;
    const seconds = (performance.now() - activeCapture.startedAt) / 1000;
    recordingTime.textContent = formatTime(seconds);
    recordingTime.setAttribute("datetime", `PT${seconds.toFixed(1)}S`);
    if (seconds >= MAXIMUM_RECORDING_SECONDS) stopRecording();
  };

  async function beginRecording(replaceIndex = null) {
    if (!supportsLiveRecording || activeCapture) return;
    showRecordingError();
    let stream = null;
    let context = null;
    let constraintsFallback = false;

    try {
      const referenceDevice = recordings[0]?.deviceId || null;
      const audioConstraints = (exact) => ({
        channelCount: { ideal: 1 },
        echoCancellation: { [exact ? "exact" : "ideal"]: false },
        noiseSuppression: { [exact ? "exact" : "ideal"]: false },
        autoGainControl: { [exact ? "exact" : "ideal"]: false },
        ...(referenceDevice ? { deviceId: { ideal: referenceDevice } } : {}),
      });
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: audioConstraints(true),
        });
      } catch (error) {
        const canRetry = ["OverconstrainedError", "NotSupportedError", "TypeError"]
          .includes(error?.name);
        if (!canRetry) throw error;
        constraintsFallback = true;
        stream = await navigator.mediaDevices.getUserMedia({
          audio: audioConstraints(false),
        });
      }
      const track = stream.getAudioTracks()[0];
      if (!track) throw new Error("No audio track was returned.");
      try {
        track.contentHint = "music";
      } catch {
        // contentHint is advisory and is not implemented by every browser.
      }
      const settings = track.getSettings?.() || {};
      const trackSettings = {
        sampleRate: trackValue(settings, "sampleRate"),
        channelCount: trackValue(settings, "channelCount"),
        sampleSize: trackValue(settings, "sampleSize"),
        latency: trackValue(settings, "latency"),
        echoCancellation: trackValue(settings, "echoCancellation"),
        noiseSuppression: trackValue(settings, "noiseSuppression"),
        autoGainControl: trackValue(settings, "autoGainControl"),
      };
      context = new AudioContextConstructor();
      await context.resume();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const silentGain = context.createGain();
      silentGain.gain.value = 0;
      const chunks = [];

      processor.onaudioprocess = (event) => {
        if (!activeCapture) return;
        const samples = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(samples.length);
        copy.set(samples);
        chunks.push(copy);
        let squareSum = 0;
        for (let index = 0; index < samples.length; index += 1) {
          squareSum += samples[index] * samples[index];
        }
        const rms = Math.sqrt(squareSum / samples.length);
        const meterDbfs = dbfs(rms);
        levelValue.style.transform = `scaleX(${Math.max(0, Math.min(1, (meterDbfs + 60) / 60))})`;
      };
      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(context.destination);

      if (sourceMode === "upload") {
        fileInput.value = "";
        takeList.replaceChildren();
        labelsFieldset.hidden = true;
        fileStatus.textContent = "No recordings selected";
        sourceMode = "record";
      }
      activeCapture = {
        stream,
        context,
        source,
        processor,
        silentGain,
        chunks,
        sampleRate: context.sampleRate,
        trackSettings,
        constraintsFallback,
        deviceId: typeof settings.deviceId === "string" ? settings.deviceId : null,
        startedAt: performance.now(),
        replaceIndex,
      };

      const takeNumber = (replaceIndex ?? recordings.length) + 1;
      recorderTitle.textContent = replaceIndex === null
        ? `Recording Take ${takeNumber}`
        : `Redoing Take ${takeNumber}`;
      recordingStatus.textContent = "Speak naturally. Stop when the sentence is complete.";
      liveRecorder.classList.add("is-recording");
      recordButton.hidden = true;
      stopButton.hidden = false;
      stopButton.focus({ preventScroll: true });
      updateReadyState();
      updateCaptureClock();
      timerId = window.setInterval(updateCaptureClock, 100);
    } catch (error) {
      activeCapture = null;
      stream?.getTracks().forEach((track) => track.stop());
      if (context && context.state !== "closed") context.close().catch(() => {});
      resetCaptureUi();
      const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
      showRecordingError(denied
        ? "Microphone access was blocked. Allow access in the browser, or use existing WAV files."
        : "The microphone could not start. Check the input device, or use existing WAV files.");
      uploadOption.open = true;
      recorderTitle.textContent = "Microphone unavailable";
      updateReadyState();
    }
  }

  recordButton.addEventListener("click", () => beginRecording());
  stopButton.addEventListener("click", stopRecording);
  protocolAck.addEventListener("change", updateReadyState);
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      sourceMode = "upload";
      clearRecordedTakes();
      recorderTitle.textContent = "Existing WAVs selected";
      recordingStatus.textContent = "Live recordings were cleared. Upload mode is active.";
    }
    renderUploadedFiles();
  });

  form.addEventListener("submit", (event) => {
    const labels = Array.from(form.querySelectorAll('input[name="labels"]'));
    const normalizedLabels = labels.map((input) => input.value.trim().toLocaleLowerCase());
    if (new Set(normalizedLabels).size !== labels.length) {
      event.preventDefault();
      labels.forEach((input) => input.setCustomValidity("Take labels must be unique."));
      labels[0]?.reportValidity();
      return;
    }
    labels.forEach((input) => input.setCustomValidity(""));
    setFileValidity();
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

  window.addEventListener("beforeunload", () => {
    if (activeCapture) activeCapture.stream.getTracks().forEach((track) => track.stop());
    recordings.forEach((recording) => URL.revokeObjectURL(recording.url));
  });

  if (!supportsLiveRecording) {
    recordButton.disabled = true;
    recorderTitle.textContent = "Live recording unavailable";
    recordingStatus.textContent = "This browser cannot capture WAV audio here. Use existing files instead.";
    uploadOption.open = true;
  }
  setFileValidity();
  updateReadyState();
}
