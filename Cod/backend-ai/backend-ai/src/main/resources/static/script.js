
const fileInput = document.getElementById('fileInput');
const resultImage = document.getElementById('resultImage');


const screenUpload = document.getElementById('screen-upload');
const screenAnalyzing = document.getElementById('screen-analyzing');
const screenResults = document.getElementById('screen-results');
const screenHistory = document.getElementById('screen-history');


function showScreen(screenElement) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    screenElement.classList.add('active');
}


document.getElementById('navToHistory').addEventListener('click', () => {
    loadHistory();
    showScreen(screenHistory);
});
document.getElementById('btnNewScan').addEventListener('click', () => showScreen(screenUpload));
document.getElementById('btnScanAgain').addEventListener('click', () => showScreen(screenUpload));
document.getElementById('btnSaveHistory').addEventListener('click', () => {
    loadHistory();
    showScreen(screenHistory);
});


fileInput.addEventListener('change', async function() {
    const file = this.files[0];
    if (!file) return;


    const reader = new FileReader();
    reader.onload = function(e) { resultImage.src = e.target.result; }
    reader.readAsDataURL(file);


    showScreen(screenAnalyzing);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch('/api/ai-detection/analiza/', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            populateResults(data);
            setTimeout(() => showScreen(screenResults), 1500);
        } else {
            alert("Eroare la procesarea pozei de către server.");
            showScreen(screenUpload);
        }
    } catch (error) {
        alert("Eroare de conexiune cu serverul.");
        showScreen(screenUpload);
    } finally {
        fileInput.value = '';
    }
});


function populateResults(data) {
    const alertBox = document.getElementById('alertBox');
    const isHealthy = data.boala_detectata.toLowerCase().includes('healthy');
    const colorVar = isHealthy ? 'var(--primary)' : 'var(--danger)';


    const today = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    document.getElementById('resultDate').innerText = today;


    if(isHealthy) {
        alertBox.className = "alert-box success";
        alertBox.innerHTML = `
            <div style="font-size: 20px;">✅</div>
            <div>
                <div class="fw-bold">No disease detected</div>
                <div style="font-size: 11px;">Healthy Plant</div>
            </div>`;
        document.getElementById('resRecommendation').innerText = "Your plant looks completely healthy. Keep up the good work with watering and nutrients!";
    } else {
        alertBox.className = "alert-box danger";
        alertBox.innerHTML = `
            <div style="font-size: 20px;">❗</div>
            <div>
                <div class="fw-bold">${data.boala_detectata} detected</div>
                <div style="font-size: 11px;">High severity · Plant disease</div>
            </div>`;
        document.getElementById('resRecommendation').innerText = "Apply specific fungicide/treatment for this disease. Remove infected leaves immediately to prevent spread.";
    }


    document.getElementById('resDiseaseName').innerText = data.boala_detectata;
    document.getElementById('resConfidence').innerText = data.siguranta.toFixed(0) + '%';
    document.getElementById('resProgressBar').style.width = data.siguranta + '%';
    document.getElementById('resProgressBar').style.backgroundColor = colorVar;
}


async function loadHistory() {
    try {
        const response = await fetch('/api/ai-detection/istoric');
        if (!response.ok) return;

        const dateIstoric = await response.json();


        const totalScans = dateIstoric.length;
        const totalDiseases = dateIstoric.filter(d => !d.boala.toLowerCase().includes('healthy')).length;
        const avgConf = totalScans > 0 ? (dateIstoric.reduce((sum, d) => sum + d.siguranta, 0) / totalScans).toFixed(0) : 0;

        document.getElementById('statTotal').innerText = totalScans;
        document.getElementById('statDiseases').innerText = totalDiseases;
        document.getElementById('statAvg').innerText = avgConf + '%';


        const listContainer = document.getElementById('historyList');
        listContainer.innerHTML = '';


        dateIstoric.reverse().forEach(item => {
            const isHealthy = item.boala.toLowerCase().includes('healthy');


            let typeClass = 'danger';
            let dotColor = 'var(--danger)';
            let iconText = '❗';

            if (isHealthy) {
                typeClass = 'success';
                dotColor = 'var(--primary)';
                iconText = '✅';
            } else if (item.siguranta < 85) {
                typeClass = 'warning';
                dotColor = 'var(--warning)';
                iconText = '⚠️';
            }


            const scanDate = new Date(item.dataScanarii).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

            const row = `
                <div class="history-item ${typeClass}">
                    <div class="history-icon" style="border: 1px solid ${dotColor}; color: ${dotColor};">${iconText}</div>
                    <div style="flex-grow: 1;">
                        <div class="fw-bold" style="font-size: 13px;">${item.boala}</div>
                        <div class="text-muted" style="font-size: 10px;">${scanDate} · ${item.siguranta.toFixed(0)}% confidence</div>
                    </div>
                    <div style="width: 8px; height: 8px; border-radius: 50%; background-color: ${dotColor};"></div>
                </div>`;
            listContainer.innerHTML += row;
        });
    } catch (error) {
        console.error("Nu s-a putut încărca istoricul.");
    }
}