const contentElements = document.querySelectorAll('.focus-left');


const options = {
    root: null, 
    rootMargin: '0px', 
    threshold: 0.5 
};


const observerCallback = (entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            
            entry.target.classList.add('visible');
            observer.unobserve(entry.target); 
        }
    });
};


const observer = new IntersectionObserver(observerCallback, options);

contentElements.forEach(element => {
    observer.observe(element);
});