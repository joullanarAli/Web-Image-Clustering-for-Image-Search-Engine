document.addEventListener("DOMContentLoaded", async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('query');

    if (query) {
        try {
            const response = await fetch(`/search?query=${encodeURIComponent(query)}`);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';

            Object.entries(data.clusters).forEach(([cluster, images]) => {
                const clusterContainer = document.createElement('div');
                clusterContainer.className = 'cluster-container';

                const clusterTitle = document.createElement('h2');
                clusterTitle.textContent = `Cluster ${cluster}`;
                clusterContainer.appendChild(clusterTitle);

                const imgContainer = document.createElement('div');
                imgContainer.className = 'image-container';

                // Display only the first 3 images from each cluster
                images.slice(0, 3).forEach(imagePath => {
                    const img = document.createElement('img');
                    img.src = `.\\clusters\\cluster_${cluster}/${imagePath}`;
                    img.alt = `Image in Cluster ${cluster}`;
                    imgContainer.appendChild(img);
                });

                clusterContainer.appendChild(imgContainer);
                resultsDiv.appendChild(clusterContainer);

                // Add a button to view all images in the cluster
                const viewAllButton = document.createElement('button');
                viewAllButton.textContent = `View All`;
                viewAllButton.onclick = () => {
                    window.location.href = `cluster.html?cluster=${cluster}`;
                };
                clusterContainer.appendChild(viewAllButton);
            });
        } catch (error) {
            console.error('Error fetching search results:', error);
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p>Sorry, there was an error retrieving the search results.</p>';
        }
    }
});

