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
    const modelNames = ['Model 1', 'Model 2', 'Model 3','Model 4'];
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


    function addMetricsToCollect() {

        const modelCards = document.querySelectorAll('.model-card .metrics');
        modelCards.forEach((card, index) => {
            const modelNumber = index + 1;
            const metricHtml = `
                <div class="divider"></div>
                <div class="metrics-container">
                
                    <div class="metric-title">Rubric Score</div>
                    <div class="rubric-score">
                        <div class="rubric-options">
                        <label><input type="radio" name="rubric-model-${modelNumber}" value="X"> X</label>
                        <label><input type="radio" name="rubric-model-${modelNumber}" value="B2"> B2</label>
                        <label><input type="radio" name="rubric-model-${modelNumber}" value="B1"> B1</label>
                        <label><input type="radio" name="rubric-model-${modelNumber}" value="C"> C</label>
                        <label><input type="radio" name="rubric-model-${modelNumber}" value="A1"> A1</label>
                        <label><input type="radio" name="rubric-model-${modelNumber}" value="A2"> A2</label>
                        </div>
                    </div>

                    <div class="metric-title">Brevity</div>
                    <div class = "rubric-score">
                        <div class="rubric-options">
                            <label><input type="radio" name="brevity-model-${modelNumber}" value="-1"> Too Concise</label>
                            <label><input type="radio" name="brevity-model-${modelNumber}" value="0"> Good</label>
                            <label><input type="radio" name="brevity-model-${modelNumber}" value="1"> Too Verbose</label>
                        </div>
                    </div>

                    <div class="metric-title">Accuracy</div>
                    <div class = "rubric-score">
                        <div class="rubric-options">
                        <label><input type="radio" name="accuracy-model-${modelNumber}" value="1"> 1</label>
                        <label><input type="radio" name="accuracy-model-${modelNumber}" value="2"> 2</label>
                        <label><input type="radio" name="accuracy-model-${modelNumber}" value="3"> 3</label>
                        <label><input type="radio" name="accuracy-model-${modelNumber}" value="4"> 4</label>
                        <label><input type="radio" name="accuracy-model-${modelNumber}" value="5"> 5</label>
                        </div>
                    </div>

                    <div class="metric-title">Is Report Dangerous?</div>
                    <div class = "rubric-score">
                        <div class="rubric-options">
                        <label><input type="radio" name="dangerous-report-model-${modelNumber}" value="1"> Yes!</label>
                        </div>
                    </div>

                </div>`;

                    // <div class="metric-title">Missed Pathology Impact</div>
                    // <div class = "rubric-score">
                    //     <div class="rubric-options">
                    //     <label><input type="radio" name="missed-pathology-model-${modelNumber}" value="1"> 1</label>
                    //     <label><input type="radio" name="missed-pathology-model-${modelNumber}" value="2"> 2</label>
                    //     <label><input type="radio" name="missed-pathology-model-${modelNumber}" value="3"> 3</label>
                    //     </div>
                    // </div>

            card.innerHTML = metricHtml;
        });
    }

    addMetricsToCollect();


    function uploadMetrics() {
        // Initialize model_metrics dictionary
        let model_metrics = {};
        
        const ranking = Array.from(modelRanking.children).map(item => item.textContent);
        // First, collect metrics for each model without considering ranking
        modelNames.forEach((modelNameID, index) => {
            const modelElement = document.querySelector(`.model-card:nth-child(${index + 1})`);
    
            const rubricInput = modelElement.querySelector(`input[name="rubric-model-${index + 1}"]:checked`);
            const brevityInput = modelElement.querySelector(`input[name="brevity-model-${index + 1}"]:checked`);
            const accuracyInput = modelElement.querySelector(`input[name="accuracy-model-${index + 1}"]:checked`);
            // const missedPathologyInput = modelElement.querySelector(`input[name="missed-pathology-model-${index + 1}"]:checked`);
            const dangerousReportInput = modelElement.querySelector(`input[name="dangerous-report-model-${index + 1}"]:checked`);

            let modelName = model_id_to_name[modelNameID];
            model_metrics[modelName] = {
                rubric: rubricInput ? rubricInput.value : null,
                brevity: brevityInput ? brevityInput.value : null,
                accuracy: accuracyInput ? accuracyInput.value : null,
                // missed_pathology: missedPathologyInput ? missedPathologyInput.value : null,
                dangerous: dangerousReportInput ? dangerousReportInput.value : null,
                rank: null // Placeholder for now, will be updated in the next step
            };
        });
    
        // Collect ranking info and update the model_metrics dictionary
        ranking.forEach((modelNameID, rank) => {
            let modelName = model_id_to_name[modelNameID];

            if (model_metrics[modelName]) {
                model_metrics[modelName].rank = rank + 1;
            }
        });
    
        const abnormal = abnormalBtn.classList.contains('btn-danger');
    
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

    function updateModelOutputs(data, show_model_name = false) {
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
                    if (show_model_name) {
                        modelHeaders[modelIndex].textContent = model_id_to_name[modelPosition]; // Set the model name
                    }
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

        // Reset any selected rubric scores by unchecking the radio buttons
        const rubricInputs = document.querySelectorAll('.rubric-options input[type="radio"]');
        rubricInputs.forEach(input => input.checked = false);

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


    // CODE TO EXECUTE ON PAGE LOAD!

    // Fetch the first image and report initially
    fetchNextImage(send_metrics_to_server = false);
});
