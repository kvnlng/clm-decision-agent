// Ask before a button marked data-confirm, and reload while a job runs.
document.addEventListener("submit", (event) => {
  const button = event.submitter;
  if (button && button.dataset.confirm && !window.confirm(button.dataset.confirm)) {
    event.preventDefault();
  }
});

if (document.querySelector("[data-job-running]")) {
  const poll = async () => {
    try {
      const response = await fetch("/jobs/current", { headers: { Accept: "application/json" } });
      const job = await response.json();
      if (!job.running) {
        window.location.reload();
        return;
      }
    } catch (error) {
      // The portal may be restarting its server; try again.
    }
    window.setTimeout(poll, 2000);
  };
  window.setTimeout(poll, 2000);
}
