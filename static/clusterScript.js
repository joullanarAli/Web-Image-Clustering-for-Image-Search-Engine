document.addEventListener("DOMContentLoaded", async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const cluster = urlParams.get('cluster');
    const itemsPerPage = 10;
    let currentPage = 1;

    if (cluster) {
        try {
            console.log(`Fetching images for cluster ${cluster}`);
            const response = await fetch(`/cluster?cluster=${encodeURIComponent(cluster)}`);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            const images = data.images;
            const totalPages = Math.ceil(images.length / itemsPerPage);

            const renderImages = (page) => {
                const start = (page - 1) * itemsPerPage;
                const end = start + itemsPerPage;
                const paginatedImages = images.slice(start, end);

                const clusterImagesDiv = document.getElementById('cluster-images');
                clusterImagesDiv.innerHTML = '';

                paginatedImages.forEach(imagePath => {
                    const img = document.createElement('img');
                    img.src = `./clusters/cluster_${cluster}/${imagePath}`;
                    img.alt = `Image in Cluster ${cluster}`;
                    clusterImagesDiv.appendChild(img);
                });
            };

            const renderPagination = () => {
                const paginationDiv = document.getElementById('pagination');
                paginationDiv.innerHTML = '';

                for (let i = 1; i <= totalPages; i++) {
                    const pageButton = document.createElement('button');
                    pageButton.textContent = i;
                    pageButton.onclick = () => {
                        currentPage = i;
                        renderImages(currentPage);
                    };
                    if (i === currentPage) {
                        pageButton.disabled = true;
                    }
                    paginationDiv.appendChild(pageButton);
                }
            };

            renderImages(currentPage);
            renderPagination();
        } catch (error) {
            console.error('Error fetching cluster images:', error);
            const clusterImagesDiv = document.getElementById('cluster-images');
            clusterImagesDiv.innerHTML = '<p>Sorry, there was an error retrieving the cluster images.</p>';
        }
    }
});
