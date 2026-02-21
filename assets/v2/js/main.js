document.addEventListener('DOMContentLoaded', () => {
    // Theme Toggle Logic
    const themeBtn = document.getElementById('theme-toggle');
    const themeIcon = themeBtn.querySelector('i');
    
    // Check for saved theme preference in localStorage, or calculate based on the clock time
    // Let's assume day is 6AM to 6PM, night is 6PM to 6AM
    const getCurrentTimeTheme = () => {
        const hour = new Date().getHours();
        return (hour >= 6 && hour < 18) ? 'light' : 'dark';
    };

    const savedTheme = localStorage.getItem('theme');
    const defaultTheme = savedTheme ? savedTheme : getCurrentTimeTheme();

    // Apply the initial theme
    document.documentElement.setAttribute('data-theme', defaultTheme);
    updateThemeIcon(defaultTheme);

    themeBtn.addEventListener('click', () => {
        let currentTheme = document.documentElement.getAttribute('data-theme');
        let newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        if (theme === 'dark') {
            themeIcon.className = 'fas fa-sun'; // Show sun to toggle to light
        } else {
            themeIcon.className = 'fas fa-moon'; // Show moon to toggle to dark
        }
    }
    
    // Animate progress bars on load
    const progressFills = document.querySelectorAll('.progress-fill');
    // Set them briefly to 0 to trigger CSS transition when set back to their defined styles
    progressFills.forEach(fill => {
        const targetWidth = fill.style.width;
        fill.style.width = '0%';
        setTimeout(() => {
            fill.style.width = targetWidth;
        }, 100);
    });
});
