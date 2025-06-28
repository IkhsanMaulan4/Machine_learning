document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const predictionForm = document.getElementById('prediction-form');
    const fileInput = document.getElementById('csv-file');

    const uploadButton = document.getElementById('upload-button');
    const predictButton = document.getElementById('predict-button');

    const uploadSpinner = uploadButton.querySelector('.spinner-border');
    const predictSpinner = predictButton.querySelector('.spinner-border');

    const uploadAlert = document.getElementById('upload-alert');
    const predictionResultDiv = document.getElementById('prediction-result');

    const predictionFieldset = document.getElementById('prediction-fieldset');
    const locationSelect = document.getElementById('location');
    const uploadedFilenameInput = document.getElementById('uploaded-filename');

    function showAlert(element, message, type = 'danger') {
        element.innerHTML = `<div class="alert alert-${type}" role="alert">${message}</div>`;
    }

    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();
        if (fileInput.files.length === 0) {
            return;
        }

        const formData = new FormData();
        formData.append('csv-file', fileInput.files[0]); 

        uploadSpinner.classList.remove('d-none');
        uploadButton.disabled = true;
        uploadAlert.innerHTML = ''; 

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (result.success) {
                showAlert(uploadAlert, 'File berhasil diupload dan model telah dilatih! Anda bisa mulai prediksi.', 'success');

                predictionFieldset.disabled = false;

                uploadedFilenameInput.value = result.filename;

                locationSelect.innerHTML = '<option selected disabled>--- Pilih Lokasi ---</option>'; // Reset
                result.locations.forEach(loc => {
                    const option = document.createElement('option');
                    option.value = loc;
                    option.textContent = loc;
                    locationSelect.appendChild(option);
                });

            } else {
                showAlert(uploadAlert, `Error: ${result.error}`);
            }
        } catch (error) {
            showAlert(uploadAlert, `Terjadi kesalahan jaringan: ${error}`);
        } finally {
            uploadSpinner.classList.add('d-none');
            uploadButton.disabled = false;
        }
    });

    predictionForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const payload = {
            filename: uploadedFilenameInput.value,
            mode: document.getElementById('mode').value,
            bed: document.getElementById('bed').value,
            bath: document.getElementById('bath').value,
            bangunan: document.getElementById('bangunan').value,
            location: document.getElementById('location').value,
            model: document.getElementById('model').value,
        };

        if (payload.location === '--- Pilih Lokasi ---') {
            predictionResultDiv.className = 'alert alert-warning fs-5';
            predictionResultDiv.textContent = 'Silakan pilih lokasi terlebih dahulu.';
            return;
        }

        predictSpinner.classList.remove('d-none');
        predictButton.disabled = true;
        predictionResultDiv.className = 'alert alert-info fs-5';
        predictionResultDiv.textContent = 'Memprediksi...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const result = await response.json();

            if (result.success) {
                predictionResultDiv.className = 'alert alert-success fs-5 fw-bold';
                predictionResultDiv.textContent = result.prediction;
            } else {
                predictionResultDiv.className = 'alert alert-danger fs-5';
                predictionResultDiv.textContent = `Error: ${result.error}`;
            }

        } catch (error) {
            predictionResultDiv.className = 'alert alert-danger fs-5';
            predictionResultDiv.textContent = `Terjadi kesalahan jaringan: ${error}`;
        } finally {
            predictSpinner.classList.add('d-none');
            predictButton.disabled = false;
        }
    });
});