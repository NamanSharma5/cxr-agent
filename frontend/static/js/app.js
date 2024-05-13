document.addEventListener('DOMContentLoaded', function() {
    const modelCards = document.querySelectorAll('.model-card');
    const rankingList = document.getElementById('model-ranking');
    const restartBtn = document.getElementById('restart-btn');
    const nextImageBtn = document.getElementById('next-image-btn');

    const modelNames = ['Model 1', 'Model 2', 'Model 3'];
    const draggableItems = [];

    // Initialize the ranking list with model names
    modelNames.forEach(name => {
        const item = document.createElement('li');
        item.textContent = name;
        item.draggable = true;
        draggableItems.push(item);
        rankingList.appendChild(item);
    });

    // Add event listeners for drag and drop functionality
    draggableItems.forEach(item => {
        item.addEventListener('dragstart', dragStart);
        item.addEventListener('dragover', dragOver);
        item.addEventListener('drop', drop);
        item.addEventListener('dragend', dragEnd);
    });

    function dragStart(e) {
        e.dataTransfer.setData('text/plain', null);
        e.currentTarget.classList.add('dragging');
    }

    function dragOver(e) {
        e.preventDefault();
    }

    function drop(e) {
        e.preventDefault();
        const draggingItem = document.querySelector('.dragging');
        const targetItem = e.currentTarget;

        if (draggingItem !== targetItem) {
            const targetIndex = Array.from(rankingList.children).indexOf(targetItem);
            rankingList.insertBefore(draggingItem, targetItem);
        }
    }

    function dragEnd(e) {
        e.currentTarget.classList.remove('dragging');
    }

    nextImageBtn.addEventListener('click', function() {
        const modelOrder = Array.from(rankingList.children).map(item => item.textContent);
        console.log('Model Order:', modelOrder);
    });

    restartBtn.addEventListener('click', function() {
        // Add code here to restart the process
        console.log('Restart button clicked');
    });
});