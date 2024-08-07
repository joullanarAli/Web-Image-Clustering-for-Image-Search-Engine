async function searchImages() {
            const query = document.getElementById('query').value;
            const response = await fetch(`/search?query=${encodeURIComponent(query)}`);
            const data = await response.json();
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';

            for (const [cluster, images] of Object.entries(data.clusters)) {
                const clusterDiv = document.createElement('div');
                clusterDiv.className = 'cluster';
                
                const title = document.createElement('div');
                title.className = 'cluster-title';
                title.textContent = `Cluster ${cluster}`;
                clusterDiv.appendChild(title);
                
                const imgContainer = document.createElement('div');
                imgContainer.className = 'image-container';
                
                images.forEach(imagePath => {
                    const img = document.createElement('img');
                    img.src = `/static/retrieved_faiss/${imagePath}`;
                    imgContainer.appendChild(img);
                });
                
                clusterDiv.appendChild(imgContainer);
                resultsDiv.appendChild(clusterDiv);
            }
        }

async function searchImages() {
    const query = document.getElementById('searchQuery').value;
    
    // Redirect to a new page with the search query as a parameter
    window.location.href = `static/results.html?query=${encodeURIComponent(query)}`;
}

function showSidebar(){
    const sidebar = document.querySelector('.sidebar')
    sidebar.style.display = 'flex'
}

function hideSidebar(){
    const sidebar = document.querySelector('.sidebar')
    sidebar.style.display = 'none'
}
