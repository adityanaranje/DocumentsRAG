document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendTrigger = document.getElementById('send-trigger');
    const uploadTrigger = document.getElementById('upload-trigger');
    const docUpload = document.getElementById('doc-upload');
    const providerSelect = document.getElementById('provider-select');
    const categorySelect = document.getElementById('category-select');
    const otherProviderGroup = document.getElementById('other-provider-group');
    const modifyGroup = document.getElementById('modify-group');
    const fileToModify = document.getElementById('file-to-modify');
    const statusBar = document.getElementById('status-bar');
    const statusText = document.getElementById('status-text');
    const clearChat = document.getElementById('clear-chat');
    const audioUpload = document.getElementById('audio-upload');
    const audioTrigger = document.getElementById('audio-trigger');

    let chatHistory = [];
    let isProcessing = false;
    let configData = null;

    // --- Dynamic UI Initialization ---
    async function loadConfig() {
        try {
            const response = await fetch('/api/config');
            configData = await response.json();
            populateProviders();
        } catch (e) { console.error("Could not load config", e); }
    }

    function populateProviders() {
        providerSelect.innerHTML = configData.providers.map(p =>
            `<option value="${p.name}">${p.name}</option>`
        ).join('');
        // Trigger initial category load
        populateCategories();
    }

    function populateCategories() {
        const selectedProviderName = providerSelect.value;
        const provider = configData.providers.find(p => p.name === selectedProviderName);
        let categories = provider ? [...provider.categories] : [];

        // Ensure "Other..." is available
        if (!categories.includes("Other...")) {
            categories.push("Other...");
        }

        categorySelect.innerHTML = categories.map(c =>
            `<option value="${c}">${c}</option>`
        ).join('');

        // Reset category input visibility
        const otherCategoryGroup = document.getElementById('other-category-group');
        if (otherCategoryGroup) {
            otherCategoryGroup.style.display = categorySelect.value === 'Other...' ? 'block' : 'none';
        }

        updateFileList();
    }

    loadConfig();

    // --- Chat Logic ---
    function addMessage(text, role) {
        if (role === 'bot') {
            typeMessage(text);
        } else {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}-message`;
            msgDiv.textContent = text;
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    }

    async function typeMessage(fullText) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        const words = fullText.split(' ');
        let currentText = '';

        for (let i = 0; i < words.length; i++) {
            currentText += words[i] + ' ';
            msgDiv.innerHTML = marked.parse(currentText + '▌'); // Add cursor effect
            chatBox.scrollTop = chatBox.scrollHeight;
            await new Promise(resolve => setTimeout(resolve, 30)); // Snappy speed
        }

        msgDiv.innerHTML = marked.parse(fullText); // Final render without cursor
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function handleChat() {
        const prompt = userInput.value.trim();
        if (!prompt || isProcessing) return;

        addMessage(prompt, 'user');
        userInput.value = '';
        isProcessing = true;

        // Add typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.textContent = 'Typing...';
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, history: chatHistory })
            });
            const data = await response.json();

            chatBox.removeChild(typingDiv);
            if (data.answer) {
                addMessage(data.answer, 'bot');
                chatHistory.push(prompt);
            } else if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            } else {
                addMessage("I'm sorry, I encountered an unknown error.", 'bot');
            }
        } catch (error) {
            chatBox.removeChild(typingDiv);
            addMessage("Connection error. Please try again later.", 'bot');
        } finally {
            isProcessing = false;
        }
    }

    sendTrigger.addEventListener('click', handleChat);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleChat();
    });

    clearChat.addEventListener('click', () => {
        chatBox.innerHTML = '<div class="message bot-message">Chat history cleared. How can I help?</div>';
        chatHistory = [];
    });

    // --- Audio Chat Logic ---
    audioTrigger.addEventListener('click', () => audioUpload.click());

    audioUpload.addEventListener('change', async () => {
        if (!audioUpload.files.length || isProcessing) return;

        const file = audioUpload.files[0];
        const audioUrl = URL.createObjectURL(file);
        const formData = new FormData();
        formData.append('audio', file);
        formData.append('history', JSON.stringify(chatHistory));

        isProcessing = true;

        // Add "Audio Uploaded" user message with Play button
        const userMsgDiv = document.createElement('div');
        userMsgDiv.className = 'message user-message audio-message-bubble';
        userMsgDiv.innerHTML = `
            <div class="audio-control">
                <button class="play-btn" id="play-${Date.now()}">
                    <i class="fas fa-play"></i>
                </button>
                <span>Audio: ${file.name}</span>
            </div>
        `;
        chatBox.appendChild(userMsgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Play functionality
        const playBtn = userMsgDiv.querySelector('.play-btn');
        const audio = new Audio(audioUrl);

        playBtn.addEventListener('click', () => {
            console.log("Play button clicked for:", file.name);
            if (audio.paused) {
                audio.play().then(() => {
                    console.log("Playback started");
                    playBtn.innerHTML = '<i class="fas fa-pause"></i>';
                }).catch(err => {
                    console.error("Playback failed:", err);
                    alert("Playback failed: " + err.message);
                });
            } else {
                audio.pause();
                playBtn.innerHTML = '<i class="fas fa-play"></i>';
            }
        });

        audio.onended = () => {
            playBtn.innerHTML = '<i class="fas fa-play"></i>';
        };

        audio.onerror = (e) => {
            console.error("Audio error event:", e);
            alert("Error loading the audio file for playback.");
        };

        // Add typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.textContent = 'Transcribing audio...';
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch('/api/audio-chat', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            chatBox.removeChild(typingDiv);
            if (data.transcription) {
                const transDiv = document.createElement('div');
                transDiv.innerHTML = `
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Raw Transcript:</div>
                    <div style="font-size: 0.85rem; font-style: italic; opacity: 0.7; margin-bottom: 12px;">"${data.transcription}"</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Refined Question:</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #818cf8;">"${data.summarized_question || data.transcription}"</div>
                `;
                transDiv.style.marginTop = '12px';
                transDiv.style.borderTop = '1px solid rgba(255,255,255,0.1)';
                transDiv.style.paddingTop = '12px';
                userMsgDiv.appendChild(transDiv);

                if (data.answer) {
                    addMessage(data.answer, 'bot');
                    chatHistory.push(data.summarized_question || data.transcription);
                }
            } else if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            }
        } catch (error) {
            if (typingDiv.parentNode) chatBox.removeChild(typingDiv);
            addMessage("Audio processing error. Please try again.", 'bot');
        } finally {
            isProcessing = false;
            audioUpload.value = '';
        }
    });

    // --- Upload and Ingestion Logic ---
    providerSelect.addEventListener('change', () => {
        otherProviderGroup.style.display = providerSelect.value === 'Other' ? 'block' : 'none';
        populateCategories();
    });

    categorySelect.addEventListener('change', () => {
        const otherCategoryGroup = document.getElementById('other-category-group');
        otherCategoryGroup.style.display = categorySelect.value === 'Other...' ? 'block' : 'none';
        updateFileList();
    });

    const otherCategoryInput = document.getElementById('other-category');
    if (otherCategoryInput) {
        otherCategoryInput.addEventListener('input', updateFileList);
    }

    const otherProviderInput = document.getElementById('other-provider');
    if (otherProviderInput) {
        otherProviderInput.addEventListener('input', updateFileList);
    }

    categorySelect.addEventListener('change', updateFileList);

    document.querySelectorAll('input[name="upload-mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            modifyGroup.style.display = e.target.value === 'Modify Existing' ? 'block' : 'none';
            if (e.target.value === 'Modify Existing') updateFileList();
        });
    });

    async function updateFileList() {
        const provider = providerSelect.value === 'Other' ? document.getElementById('other-provider').value : providerSelect.value;
        const category = categorySelect.value === 'Other...' ? document.getElementById('other-category').value : categorySelect.value;

        if (!provider || !category || category === '') {
            fileToModify.innerHTML = '<option value="">(Waiting for selection...)</option>';
            return;
        }

        try {
            const response = await fetch(`/api/files?provider=${encodeURIComponent(provider)}&category=${encodeURIComponent(category)}`);
            const data = await response.json();
            fileToModify.innerHTML = '<option value="">Select file to replace...</option>';
            data.files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f;
                opt.textContent = f;
                fileToModify.appendChild(opt);
            });
        } catch (e) { console.error("Could not fetch files", e); }
    }

    uploadTrigger.addEventListener('click', () => docUpload.click());

    docUpload.addEventListener('change', async () => {
        if (!docUpload.files.length) return;

        const formData = new FormData();
        const provider = providerSelect.value === 'Other' ? document.getElementById('other-provider').value : providerSelect.value;
        const category = categorySelect.value === 'Other...' ? document.getElementById('other-category').value : categorySelect.value;
        const mode = document.querySelector('input[name="upload-mode"]:checked').value;

        if (!provider || !category) {
            alert("Please specify both Provider and Category.");
            return;
        }

        formData.append('file', docUpload.files[0]);
        formData.append('provider', provider);
        formData.append('category', category);
        formData.append('mode', mode);
        if (mode === 'Modify Existing') {
            formData.append('file_to_modify', fileToModify.value);
        }

        uploadTrigger.disabled = true;
        uploadTrigger.textContent = 'Uploading...';

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.error) alert(data.error);
            else {
                // Refresh config to include new provider/category if added
                await loadConfig();
                // Start polling status
                pollStatus();
            }
        } catch (e) {
            alert("Upload failed.");
        } finally {
            uploadTrigger.disabled = false;
            uploadTrigger.textContent = 'Choose & Process';
            docUpload.value = '';
        }
    });

    function pollStatus() {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                statusText.textContent = data.status;
                statusBar.style.width = data.progress + '%';

                if (data.status === 'Completed Successfully!' || data.status.startsWith('Failed')) {
                    clearInterval(interval);
                    if (data.status === 'Completed Successfully!') {
                        statusText.style.color = '#10b981'; // Green
                    } else {
                        statusText.style.color = '#ef4444'; // Red
                    }
                }
            } catch (e) {
                clearInterval(interval);
            }
        }, 1000);
    }
});
