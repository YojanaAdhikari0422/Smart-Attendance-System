"use strict";

const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";
const captureWidth = 640;
const sampleCount = 6;

function frameFrom(video) {
  if (!video || !video.videoWidth || !video.videoHeight) {
    throw new Error("Camera is not ready yet. Please wait a moment and try again.");
  }
  const scale = Math.min(1, captureWidth / video.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.72);
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken },
    body: JSON.stringify(body),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(result.error || `Request failed (${response.status}).`);
  }
  return result;
}

function stopStream(video) {
  const stream = video?.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }
}

const liveVideo = document.querySelector("#camera-video");
if (liveVideo) {
  const startButton = document.querySelector("#start-camera");
  const stopButton = document.querySelector("#stop-camera");
  const placeholder = document.querySelector("#camera-placeholder");
  const stateLabel = document.querySelector("#camera-state");
  const resultPanel = document.querySelector("#recognition-result");
  let stream = null;
  let isRunning = false;
  let isProcessing = false;
  let timer = null;

  function showRecognition(state, title, message) {
    const icon = state === "recognized" ? "✓" : state === "error" ? "!" : "◷";
    resultPanel.className = `recognition-result result-${state}`;
    resultPanel.replaceChildren();
    const badge = document.createElement("span");
    badge.className = "result-icon";
    badge.textContent = icon;
    const heading = document.createElement("strong");
    heading.textContent = title;
    const detail = document.createElement("p");
    detail.textContent = message;
    resultPanel.append(badge, heading, detail);
  }

  function stopLiveCamera() {
    isRunning = false;
    isProcessing = false;
    window.clearTimeout(timer);
    stopStream(liveVideo);
    stream = null;
    liveVideo.hidden = true;
    placeholder.hidden = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    stateLabel.textContent = "Camera stopped";
    showRecognition("idle", "Waiting for camera", "Start the camera to begin.");
  }

  async function pollRecognition() {
    if (!isRunning) return;
    if (!document.hidden && !isProcessing) {
      isProcessing = true;
      try {
        const result = await postJson("/api/recognize", { frame: frameFrom(liveVideo) });
        if (result.state === "recognized") {
          showRecognition("recognized", result.name, result.message);
          stateLabel.textContent = "Person recognized";
        } else if (result.state === "unknown") {
          showRecognition("unknown", "Not recognized", result.message);
          stateLabel.textContent = "Looking for a registered person";
        } else {
          showRecognition("idle", "Looking for a face", result.message);
          stateLabel.textContent = "Camera active";
        }
      } catch (error) {
        showRecognition("error", "Recognition unavailable", error.message);
        stateLabel.textContent = "Connection issue";
      } finally {
        isProcessing = false;
      }
    }
    if (isRunning) timer = window.setTimeout(pollRecognition, 1500);
  }

  startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    stateLabel.textContent = "Requesting camera access";
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires HTTPS or localhost and a supported browser.");
      }
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      liveVideo.srcObject = stream;
      await liveVideo.play();
      liveVideo.hidden = false;
      placeholder.hidden = true;
      isRunning = true;
      stopButton.disabled = false;
      stateLabel.textContent = "Camera active";
      showRecognition("idle", "Looking for a face", "Center one face in the camera view.");
      pollRecognition();
    } catch (error) {
      stopStream(liveVideo);
      stream = null;
      startButton.disabled = false;
      stopButton.disabled = true;
      stateLabel.textContent = "Camera unavailable";
      showRecognition("error", "Could not start camera", error.message);
    }
  });
  stopButton.addEventListener("click", stopLiveCamera);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && isRunning) {
      stopLiveCamera();
    }
  });
  window.addEventListener("pagehide", stopLiveCamera);
}

const enrollVideo = document.querySelector("#enroll-video");
if (enrollVideo) {
  const startButton = document.querySelector("#enroll-start-camera");
  const captureButton = document.querySelector("#capture-samples");
  const saveButton = document.querySelector("#save-person");
  const placeholder = document.querySelector("#enroll-placeholder");
  const progress = document.querySelector("#capture-progress");
  const feedback = document.querySelector("#enroll-feedback");
  const nameInput = document.querySelector("#enroll-name");
  const idInput = document.querySelector("#enroll-id");
  const consentInput = document.querySelector("#enroll-consent");
  const dialog = document.querySelector("#enroll-dialog");
  let frames = [];
  let captureStream = null;
  let isCapturing = false;

  function updateProgress(message = "") {
    progress.textContent = `${frames.length} of ${sampleCount} frames captured`;
    if (message) feedback.textContent = message;
    captureButton.disabled = !captureStream || isCapturing || frames.length >= sampleCount;
    saveButton.disabled = frames.length < 3 || isCapturing;
  }

  function stopCaptureCamera() {
    isCapturing = false;
    stopStream(enrollVideo);
    captureStream = null;
    enrollVideo.hidden = true;
    placeholder.hidden = false;
    startButton.disabled = false;
    updateProgress();
  }

  window.stopEnrollmentCamera = stopCaptureCamera;

  startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    feedback.textContent = "";
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires HTTPS or localhost and a supported browser.");
      }
      captureStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      enrollVideo.srcObject = captureStream;
      await enrollVideo.play();
      enrollVideo.hidden = false;
      placeholder.hidden = true;
      frames = [];
      updateProgress();
    } catch (error) {
      captureStream = null;
      startButton.disabled = false;
      feedback.textContent = error.message;
    }
  });

  captureButton.addEventListener("click", async () => {
    if (!captureStream) return;
    isCapturing = true;
    frames = [];
    updateProgress("Keep your face centered while the samples are captured.");
    try {
      for (let index = 0; index < sampleCount; index += 1) {
        if (!captureStream.active) throw new Error("Camera stopped. Start it again to capture samples.");
        frames.push(frameFrom(enrollVideo));
        updateProgress();
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
      updateProgress("Samples captured. Review the consent checkbox and save the profile.");
    } catch (error) {
      frames = [];
      updateProgress(error.message);
    } finally {
      isCapturing = false;
      updateProgress(feedback.textContent);
    }
  });

  saveButton.addEventListener("click", async () => {
    const name = nameInput.value.trim();
    const studentId = idInput.value.trim();
    if (!name || !studentId) {
      updateProgress("Enter a name and student ID.");
      return;
    }
    if (!consentInput.checked) {
      updateProgress("Confirm that you have permission before enrolling this person.");
      return;
    }
    if (frames.length < 3) {
      updateProgress("Capture at least three frames before saving.");
      return;
    }

    saveButton.disabled = true;
    captureButton.disabled = true;
    startButton.disabled = true;
    feedback.textContent = "Processing frames securely...";
    try {
      const result = await postJson("/api/people", {
        name,
        student_id: studentId,
        frames,
        consent: true,
      });
      frames = [];
      stopCaptureCamera();
      dialog.close();
      window.location.assign(`/people?enrolled=${encodeURIComponent(result.student_id)}`);
    } catch (error) {
      feedback.textContent = error.message;
      isCapturing = false;
      updateProgress(error.message);
    }
  });

  dialog.addEventListener("close", () => {
    frames = [];
    updateProgress();
  });
}
