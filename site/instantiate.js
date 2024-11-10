const itemTemplate = document.getElementById("item-template");
const container = document.getElementById("item-container");

for (let i=0; i<5; i++){

	console.log(i);
	const newItem = itemTemplate.content.cloneNode(true);
	const titleAdd = newItem.querySelector(".item-title");
	titleAdd.textContent = titleAdd.textContent + " " + (i+1);
	
	
	container.appendChild(newItem);
}


