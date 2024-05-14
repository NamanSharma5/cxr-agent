document.addEventListener('DOMContentLoaded', function() {
    const modelRanking = document.getElementById('model-ranking');
    const restartBtn = document.getElementById('restart-btn');
    const nextImageBtn = document.getElementById('next-image-btn');
    const imageContainer = document.querySelector('.image-container img');
    const referenceReport = document.getElementById('reference-report');

    const modelNames = ['Model 1', 'Model 2', 'Model 3'];

    // Initialize the ranking list with model names
    modelNames.forEach(name => {
        const item = document.createElement('li');
        item.textContent = name;
        item.classList.add('list-group-item');
        modelRanking.appendChild(item);
    });

    // Instantiate SortableJS on the ranking list
    Sortable.create(modelRanking, {
        animation: 150,
        ghostClass: 'sortable-ghost'
    });

    function updateImageAndReport(data) {
        if (data.image_path) {
            imageContainer.src = `static${data.image_path}`;
        }
        referenceReport.value = data.report;
    }

    function fetchNextImage() {
        fetch('/next_image')
            .then(response => response.json())
            .then(data => updateImageAndReport(data))
            .catch(error => console.error('Error fetching the next image:', error));
    }

    nextImageBtn.addEventListener('click', fetchNextImage);

    restartBtn.addEventListener('click', function() {
        fetch('/restart', { method: 'POST' })
            .then(response => {
                if (response.ok) {
                    fetchNextImage();
                } else {
                    console.error('Error restarting the process.');
                }
            })
            .catch(error => console.error('Error restarting the process:', error));
    });

});