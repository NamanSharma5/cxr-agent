let currentSubject = null;
let model_name_to_id = null;
let model_id_to_name = null;

document.addEventListener('DOMContentLoaded', function() {
    const modelRanking = document.getElementById('model-ranking');
    const restartBtn = document.getElementById('restart-btn');
    const nextImageBtn = document.getElementById('next-image-btn');
    const imageContainer = document.querySelector('.image-container img');
    const referenceReport = document.getElementById('reference-report');
    const modelCards = document.querySelectorAll('.model-card .card-body');
    const modelHeaders = document.querySelectorAll('.model-card .card-header');
    const abnormalBtn = document.getElementById('abnormal-btn');


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

    // Add metric options dynamically under each model card
    // modelCards.forEach((card, index) => {
    //     const metricsDiv = card.querySelector('.metrics');
    //     const modelNumber = index + 1;
    //     const metricHtml = `
    //         <div class="rubric-score">
    //             <label>Rubric score:</label>
    //             <div class="rubric-options">
    //                 <label><input type="radio" name="rubric-model-${modelNumber}" value="A2"> A2</label>
    //                 <label><input type="radio" name="rubric-model-${modelNumber}" value="A1"> A1</label>
    //                 <label><input type="radio" name="rubric-model-${modelNumber}" value="C"> C</label>
    //                 <label><input type="radio" name="rubric-model-${modelNumber}" value="B1"> B1</label>
    //                 <label><input type="radio" name="rubric-model-${modelNumber}" value="B2"> B2</label>
    //                 <label><input type="radio" name="rubric-model-${modelNumber}" value="X"> X</label>
    //             </div>
    //         </div>`;
    //     metricsDiv.innerHTML = metricHtml;
    // });

    function uploadMetrics() {
        const ranking = Array.from(modelRanking.children).map(item => item.textContent);
        const abnormal = abnormalBtn.classList.contains('btn-danger');
        
        let model_metrics = {};

        // Populate model metrics
        ranking.forEach((model_id, index) => {
            model_name = model_id_to_name[model_id];
            model_metrics[model_name] = {
                rank: index + 1
            };
        });

        // Add metadata information
        const metadata = {
            subject: currentSubject,
            abnormal: abnormal
        };

        model_metrics['metadata'] = metadata;

        // Logging for debugging, replace with API call or other logic as needed
        console.log('Model Metrics:', model_metrics);

        fetch('/upload_metrics', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(model_metrics)
        })
        .then(response => {
            if (response.ok) {
                console.log('Metrics uploaded successfully.');
            } else {
                console.error('Error uploading metrics.');
            }
        })
        .catch(error => console.error('Error uploading metrics:', error));
    }

    function updateImageAndReport(data) {
        if (data.image_data) {
            imageContainer.src = `data:image/jpeg;base64,${data.image_data}`;
        }
        referenceReport.textContent = data.report; // Set the text content of the div
        currentSubject = data.subject;

        // Fetch model outputs asynchronously
        fetch('/get_model_outputs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ subject: data.subject })
        })
        .then(response => response.json())
        .then(outputData => updateModelOutputs(outputData))
        .catch(error => console.error('Error fetching model outputs:', error));
    }

    function updateModelOutputs(data) {
        model_name_to_id = data.model_name_to_id; // store the model name to id mapping
        model_id_to_name = data.model_id_to_name; // store the model id to name mapping
        
        // Update model outputs
        if (data.model_outputs) {
            const model_name_to_id = data.model_name_to_id;
            Object.keys(data.model_outputs).forEach(modelKey => {
                const modelPosition = model_name_to_id[modelKey];
                const modelIndex = modelNames.indexOf(modelPosition);
                if (modelIndex !== -1) {
                    modelHeaders[modelIndex].textContent = modelPosition; // Set the model name
                    modelCards[modelIndex].textContent = data.model_outputs[modelKey]; // Set the model output
                }
            });
        }
    }

    function fetchNextImage(send_metrics_to_server = true) {
        if (send_metrics_to_server) {
            uploadMetrics();
        }
        // first clear the model outputs (i.e. the content of model cards)
        modelCards.forEach(card => card.textContent = '');
        fetch('/next_image')
            .then(response => response.json())
            .then(data => updateImageAndReport(data))
            .catch(error => console.error('Error fetching the next image:', error));
    }
    

    abnormalBtn.addEventListener('click', function() {
      if (abnormalBtn.classList.contains('btn-danger')) {
        abnormalBtn.classList.remove('btn-danger');
        abnormalBtn.classList.add('btn-light');
        abnormalBtn.classList.add('text-danger');
        abnormalBtn.textContent = 'Normal';
      } else {
        abnormalBtn.classList.remove('btn-light');
        abnormalBtn.classList.remove('text-danger');
        abnormalBtn.classList.add('btn-danger');
        abnormalBtn.textContent = 'Abnormal';
      }
    });

    nextImageBtn.addEventListener('click', function() {
        nextImageBtn.disabled = true; // Disable the button
        fetchNextImage();
        nextImageBtn.disabled = false; // Enable the button after fetchNextImage has finished
    });

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
    fetchNextImage(send_metrics_to_server = false);
});
