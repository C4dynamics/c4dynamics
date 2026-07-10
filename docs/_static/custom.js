document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
});