/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0.
 */

(function () {
  "use strict";

  const root = document.documentElement;
  const themeButton = document.querySelector("[data-theme-toggle]");
  const storedTheme = localStorage.getItem("vindex-docs-theme");
  if (storedTheme === "dark" || storedTheme === "light") {
    root.dataset.theme = storedTheme;
  } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.dataset.theme = "dark";
  }

  function syncThemeButton() {
    if (!themeButton) return;
    const dark = root.dataset.theme === "dark";
    themeButton.textContent = dark ? "☀" : "◐";
    themeButton.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    themeButton.title = dark ? "Light mode" : "Dark mode";
  }

  if (themeButton) {
    syncThemeButton();
    themeButton.addEventListener("click", function () {
      root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
      localStorage.setItem("vindex-docs-theme", root.dataset.theme);
      syncThemeButton();
    });
  }

  const navButton = document.querySelector("[data-nav-toggle]");
  const nav = document.querySelector("[data-site-nav]");
  if (navButton && nav) {
    navButton.addEventListener("click", function () {
      const open = nav.classList.toggle("open");
      navButton.setAttribute("aria-expanded", String(open));
    });
    nav.addEventListener("click", function () {
      nav.classList.remove("open");
      navButton.setAttribute("aria-expanded", "false");
    });
  }

  document.querySelectorAll(".code-block").forEach(function (block) {
    const code = block.querySelector("code");
    if (!code) return;
    const button = document.createElement("button");
    button.className = "copy-button";
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code");
    button.addEventListener("click", async function () {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
      } catch (_) {
        button.textContent = "Copy failed";
      }
      window.setTimeout(function () {
        button.textContent = "Copy";
      }, 1400);
    });
    block.appendChild(button);
  });

  const tocLinks = Array.from(document.querySelectorAll(".toc a[href^='#']"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    const sectionById = new Map(tocLinks.map(function (link) {
      return [link.getAttribute("href").slice(1), link];
    }));
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        tocLinks.forEach(function (link) { link.classList.remove("active"); });
        const link = sectionById.get(entry.target.id);
        if (link) link.classList.add("active");
      });
    }, { rootMargin: "-18% 0px -68% 0px" });
    sectionById.forEach(function (_, id) {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  }

  const filterButtons = document.querySelectorAll("[data-filter]");
  const comparisonRows = document.querySelectorAll("[data-index-row]");
  filterButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      const filter = button.dataset.filter;
      filterButtons.forEach(function (candidate) {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      comparisonRows.forEach(function (row) {
        row.classList.toggle("is-hidden", filter !== "all" && !row.dataset.fit.split(" ").includes(filter));
      });
    });
  });
})();
