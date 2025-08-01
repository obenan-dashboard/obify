// Main JavaScript for Obify application

document.addEventListener('DOMContentLoaded', function() {
    // Documentation modal handler
    const showDocsButton = document.getElementById('showDocsButton');
    if (showDocsButton) {
        showDocsButton.addEventListener('click', function() {
            const docsModal = new bootstrap.Modal(document.getElementById('docsModal'));
            docsModal.show();
        });
    }
    
    // File upload area highlighting
    const uploadArea = document.getElementById('uploadArea');
    const uploadForm = document.querySelector('.upload-form');
    const fileInput = document.getElementById('fileUpload');
    
    if (uploadArea) {
        // Handle drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle drop functionality
        uploadArea.addEventListener('drop', handleDrop, false);
        
        // No need for click handler - the file input already covers the entire area
        // and will naturally trigger the file dialog when clicked

        function highlight(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.add('highlight');
        }

        function unhighlight(e) {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('highlight');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const dt = e.dataTransfer;
            if (dt.files.length) {
                fileInput.files = dt.files;
                // Trigger the change event
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        }
    }
    
    // Show file name when selected
    const uploadButton = document.querySelector('button[type="submit"]');
    const uploadPreview = document.getElementById('uploadPreview');
    
    if (fileInput && uploadButton && uploadPreview) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const fileSize = (file.size / (1024 * 1024)).toFixed(2); // Convert to MB
                const fileExtension = file.name.split('.').pop().toLowerCase();
                
                if (['csv', 'txt', 'xlsx', 'xls'].includes(fileExtension)) {
                    // Valid file
                    uploadPreview.innerHTML = `
                        <i class="bi bi-file-earmark-text display-4 mb-3 text-success"></i>
                        <h4>${file.name}</h4>
                        <p class="text-muted">${fileSize} MB</p>
                    `;
                    uploadButton.disabled = false;
                } else {
                    // Invalid file type
                    uploadPreview.innerHTML = `
                        <i class="bi bi-exclamation-circle display-4 mb-3 text-danger"></i>
                        <h4>Invalid File Type</h4>
                        <p class="text-muted">Please select CSV, TXT, XLS or XLSX</p>
                    `;
                    uploadButton.disabled = true;
                }
            }
        });
    }
    
    // Form validation
    const configForm = document.getElementById('configForm');
    if (configForm) {
        configForm.addEventListener('submit', function(event) {
            console.log('Form is being submitted');
            console.log('Selected models:', document.querySelectorAll('input[name="models"]:checked'));
            console.log('Prompt template:', document.getElementById('promptTemplate').value);
            
            const selectedModels = document.querySelectorAll('input[name="models"]:checked');
            if (selectedModels.length === 0) {
                event.preventDefault();
                console.error('No models selected!');
                alert('Please select at least one model for analysis.');
                return;
            }
            
            const promptTemplate = document.getElementById('promptTemplate').value;
            if (!promptTemplate.includes('{review}')) {
                event.preventDefault();
                console.error('No {review} placeholder in template!');
                alert('Prompt template must include {review} placeholder.');
                return;
            }
            
            // Log what we're about to submit
            console.log('Form validation passed, preparing to submit');
            const formData = new FormData(configForm);
            for (let [key, value] of formData.entries()) {
                console.log(`Form data: ${key} = ${value.substring(0, 100)}${value.length > 100 ? '...' : ''}`);
            }
            
            // Show processing modal
            console.log('Showing processing modal');
            const processingModal = new bootstrap.Modal(document.getElementById('processingModal'));
            processingModal.show();
            document.getElementById('runButton').disabled = true;
            
            console.log('Submitting form to server...');
            return true;
        });
    }
    
    // Copy JSON response to clipboard
    const copyButtons = document.querySelectorAll('.copy-response');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const responseText = this.getAttribute('data-response');
            navigator.clipboard.writeText(responseText).then(() => {
                // Show copied tooltip
                this.setAttribute('data-original-title', 'Copied!');
                const tooltip = bootstrap.Tooltip.getInstance(this);
                if (tooltip) {
                    tooltip.show();
                    setTimeout(() => tooltip.hide(), 1000);
                }
            }).catch(err => {
                console.error('Could not copy text: ', err);
            });
        });
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
