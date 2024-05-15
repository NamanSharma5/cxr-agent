document.addEventListener('DOMContentLoaded', function() {
    const modelRanking = document.getElementById('model-ranking');
    const restartBtn = document.getElementById('restart-btn');
    const nextImageBtn = document.getElementById('next-image-btn');
    const imageContainer = document.querySelector('.image-container img');
    const referenceReport = document.getElementById('reference-report');
    const modelCards = document.querySelectorAll('.model-card .card-body');
    const modelHeaders = document.querySelectorAll('.model-card .card-header');

    // Initialize the ranking list with model names
    const modelNames = ['Model 1', 'Model 2', 'Model 3'];
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
        if (data.image_data) {
            imageContainer.src = `data:image/jpeg;base64,${data.image_data}`;
        }
        referenceReport.textContent = data.report; // Set the text content of the div

        // Fetch model outputs asynchronously if not included
        if (!data.model_outputs) {
            fetch('/get_model_outputs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject: data.subject })
            })
            .then(response => response.json())
            .then(outputData => updateModelOutputs(outputData))
            .catch(error => console.error('Error fetching model outputs:', error));
        } else {
            updateModelOutputs({
                model_outputs: data.model_outputs,
                model_mapping: data.model_mapping
            });
        }
    }

    function updateModelOutputs(data) {
        // Update model outputs
        if (data.model_outputs) {
            const modelMapping = data.model_mapping;
            Object.keys(data.model_outputs).forEach(modelKey => {
                const modelPosition = modelMapping[modelKey];
                const modelIndex = modelNames.indexOf(modelPosition);
                if (modelIndex !== -1) {
                    modelHeaders[modelIndex].textContent = modelPosition; // Set the model name
                    modelCards[modelIndex].textContent = data.model_outputs[modelKey]; // Set the model output
                }
            });
        }
    }

    function fetchNextImage() {
        // first clear the model outputs (i.e. the content of model cards)
        modelCards.forEach(card => card.textContent = '');
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

    // Fetch the first image and report initially
    fetchNextImage();
});