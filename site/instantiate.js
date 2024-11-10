document.addEventListener("DOMContentLoaded", function () {
    // Local JSON data for testing
    const foodData = [
        { "name": "Apple", "sustainability_score": 5, "times_eaten": 15 },
        { "name": "Orange", "sustainability_score": 1, "times_eaten": 4 },
        { "name": "Cereal", "sustainability_score": 3, "times_eaten": 4 },
        { "name": "French Fries", "sustainability_score": 4, "times_eaten": 10 }
    ];

    const itemTemplate = document.getElementById("item-template");
    const container = document.getElementById("item-container");

    // Iterate over each item in the local data
    foodData.forEach((item, index) => {
        if (index < 5) { // Limit to 5 items if necessary
            const newItem = itemTemplate.content.cloneNode(true); // Clone the template structure

            // Populate each field within the template
            const titleAdd = newItem.querySelector(".item-title");
            const typeOfFoodCell = newItem.querySelector("[data-table-cell='Type of Food: []']");
            const reduceCell = newItem.querySelector("[data-table-cell='How much to reduce: []']");
            const sustainabilityGradeCell = newItem.querySelector("[data-table-cell='Sustainability Grade: []']");

            // Set text content for each field in the cloned template
            titleAdd.textContent = `Item: ${item.name}`;
            typeOfFoodCell.textContent = `Type of Food: ${item.name}`;
            reduceCell.textContent = `How much to reduce: ${item.times_eaten} times`; // Example based on times_eaten
            sustainabilityGradeCell.textContent = `Sustainability Grade: ${item.sustainability_score}`;

            // Append the populated clone to the container
            container.appendChild(newItem);
        }
    });
});
