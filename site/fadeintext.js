function fadeIn(){
	const textElement = document.getElementById("big-text");
	
	textElement.classList.add("visible");

}

function fadeInArrow(){
	const textElement = document.getElementById("arrow");
	
	textElement.classList.add("visible");

}

window.onload = function(){
	fadeIn();
	fadeInArrow();
	
	
}