(() => {
  const body = document.body;
  const progress = document.querySelector('.progress span');
  const themeToggle = document.querySelector('[data-theme-toggle]');
  const savedTheme = localStorage.getItem('vllm-reader-theme');

  if (savedTheme === 'dark') body.classList.add('dark');
  themeToggle?.addEventListener('click', () => {
    body.classList.toggle('dark');
    localStorage.setItem('vllm-reader-theme', body.classList.contains('dark') ? 'dark' : 'light');
  });

  const updateProgress = () => {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0}%`;
  };
  window.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
})();
