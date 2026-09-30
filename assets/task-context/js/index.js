document.addEventListener('DOMContentLoaded', function () {
  bulmaCarousel.attach('#results-carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: false,
    autoplay: false
  });

  var controls = document.querySelectorAll(
    '#results-carousel .slider-navigation-previous, ' +
    '#results-carousel .slider-navigation-next, ' +
    '#results-carousel .slider-page'
  );
  controls.forEach(function (control) {
    control.setAttribute('role', 'button');
    control.setAttribute('tabindex', '0');
    if (control.classList.contains('slider-navigation-previous')) {
      control.setAttribute('aria-label', 'Previous figure');
    } else if (control.classList.contains('slider-navigation-next')) {
      control.setAttribute('aria-label', 'Next figure');
    } else {
      var pages = Array.from(control.parentElement.children);
      control.setAttribute('aria-label', 'Show figure ' + (pages.indexOf(control) + 1));
    }
    control.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        event.stopPropagation();
        control.click();
      }
    });
  });
});
