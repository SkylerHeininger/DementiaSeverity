(function () {
  const page = JSON.parse(document.getElementById("visit-data").textContent);
  const base = `/visits/${page.visit_id}`;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const button = $("process-btn");

  function renderSteps(steps) {
    $("steps").innerHTML = Object.values(steps).map((s) => {
      const timed = s.seconds != null && s.status !== "running" && s.status !== "pending";
      return `<li><div class="step-top"><span>${esc(s.label)}</span><span class="pill ${s.status}">${s.status}${timed ? ` &middot; ${s.seconds}s` : ""}</span></div>` +
        `${s.message ? `<span class="error-text">${esc(s.message)}</span>` : ""}</li>`;
    }).join("");
  }

  function setBusy(busy) {
    button.disabled = busy;
    if (busy) button.textContent = "Processing...";
  }

  async function poll() {
    const status = await (await fetch(`${base}/status`)).json();
    renderSteps(status.steps);
    if (status.running) return setTimeout(poll, 1000);
    if (status.finished) return location.reload();
    setBusy(false);
    button.textContent = "Process again";
  }

  async function start() {
    setBusy(true);
    await fetch(`${base}/process`, { method: "POST" });
    poll();
  }

  const TABS = ["stats", "da", "rep", "sev"];
  const selected = new Set();

  function highlight(match, scroll) {
    const turns = [...document.querySelectorAll(".turn")];
    turns.forEach((t) => t.classList.toggle("hl", match(t)));
    const first = turns.find((t) => t.classList.contains("hl"));
    if (scroll && first) {
      const box = $("transcript");
      box.scrollBy({ top: first.getBoundingClientRect().top - box.getBoundingClientRect().top - box.clientHeight / 3, behavior: "smooth" });
    }
  }

  const highlightSelected = (scroll) => highlight((t) => selected.size > 0 && t.dataset.groups.split(" ").some((g) => selected.has(g)), scroll);

  function selectTab(name, focus) {
    document.querySelectorAll(".tab").forEach((tab) => {
      const on = tab.dataset.tab === name;
      tab.classList.toggle("active", on);
      tab.setAttribute("aria-selected", on);
      tab.tabIndex = on ? 0 : -1;
      if (on && focus) tab.focus();
    });
    TABS.forEach((n) => { $(`panel-${n}`).hidden = n !== name; });
    history.replaceState(null, "", `#${name}`);
    if (name === "da") highlightSelected(false); else highlight(() => false, false);
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      selectTab(TABS[(TABS.indexOf(tab.dataset.tab) + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length], true);
    });
  });

  document.querySelectorAll(".gchip").forEach((chip) => chip.addEventListener("click", () => {
    const group = chip.dataset.group;
    selected.has(group) ? selected.delete(group) : selected.add(group);
    chip.classList.toggle("on");
    highlightSelected(true);
  }));

  document.querySelectorAll(".seg-btn").forEach((btn) => btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".sev-view").forEach((v) => { v.hidden = v.dataset.view !== btn.dataset.view; });
  }));

  button.addEventListener("click", start);
  renderSteps(page.steps);
  selectTab(TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "stats");
  if (page.running) { setBusy(true); poll(); }
})();
