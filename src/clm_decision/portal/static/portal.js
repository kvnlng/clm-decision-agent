// Ask before a button marked data-confirm; while a job runs, poll, then go to the page it names (a GET, so no form
// is sent again).
document.addEventListener("submit", (event) => {
  const button = event.submitter;
  if (button && button.dataset.confirm && !window.confirm(button.dataset.confirm)) {
    event.preventDefault();
  }
});

const running = document.querySelector("[data-job-running]");
if (running) {
  const poll = async () => {
    try {
      const response = await fetch("/jobs/current", { headers: { Accept: "application/json" } });
      const job = await response.json();
      if (!job.running) {
        window.location.replace(running.dataset.jobRunning);
        return;
      }
    } catch (error) {
      // The portal may be restarting its server; try again.
    }
    window.setTimeout(poll, 2000);
  };
  window.setTimeout(poll, 2000);
}
