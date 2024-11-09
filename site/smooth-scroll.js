let scrollSpeed = 7; 

window.addEventListener('wheel', function(event) {
    event.preventDefault();

    let delta = event.deltaY || event.detail || event.wheelDelta;
    if (delta > 0) {
        window.scrollBy(0, scrollSpeed); 
    } else {
        window.scrollBy(0, -scrollSpeed); 
    }
}, { passive: false });