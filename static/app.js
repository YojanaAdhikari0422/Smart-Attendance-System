"use strict";

document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!window.confirm(form.dataset.confirm)) {
      event.preventDefault();
    }
  });
});

const enrollDialog = document.querySelector("#enroll-dialog");
document.querySelectorAll("[data-open-enroll]").forEach((button) => {
  button.addEventListener("click", () => {
    if (enrollDialog && !enrollDialog.open) {
      enrollDialog.showModal();
    }
  });
});

if (enrollDialog) {
  enrollDialog.addEventListener("close", () => {
    if (window.stopEnrollmentCamera) {
      window.stopEnrollmentCamera();
    }
  });
}
