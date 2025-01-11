// Store prompts by message ID
const messagePrompts = new Map();

function addMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    // Generate message ID if not provided
    const messageId = message.message_id || crypto.randomUUID();
    messageDiv.dataset.messageId = messageId;
    
    // Store prompt details if available
    if (message.prompt) {
        messagePrompts.set(messageId, message.prompt);
    }
    
    // Add message content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatMessage(message.response || message);
    messageDiv.appendChild(contentDiv);
    
    // Add prompt details button if available
    if (message.prompt) {
        const detailsButton = document.createElement('button');
        detailsButton.className = 'prompt-details-button';
        detailsButton.textContent = 'Show Details';
        detailsButton.onclick = () => showPromptDetails(messageId);
        messageDiv.appendChild(detailsButton);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showPromptDetails(messageId) {
    const promptDetails = messagePrompts.get(messageId);
    if (!promptDetails) return;
    
    try {
        const details = JSON.parse(promptDetails);
        const detailsDiv = document.getElementById('prompt-details');
        detailsDiv.innerHTML = formatPromptDetails(details);
        detailsDiv.style.display = 'block';
    } catch (e) {
        console.error('Error parsing prompt details:', e);
    }
}

function formatPromptDetails(details) {
    let html = '<div class="prompt-details-content">';
    
    // Query Analysis
    if (details["Query Analysis"]) {
        html += '<div class="details-section"><h3>Query Analysis</h3>';
        html += formatObject(details["Query Analysis"]);
        html += '</div>';
    }
    
    // Memory Search
    if (details["Memory Search"]) {
        html += '<div class="details-section"><h3>Memory Search</h3>';
        html += formatObject(details["Memory Search"]);
        
        // Display retrieved memories if available
        if (details["Memory Search"]["Retrieved Memories"]) {
            html += '<h4>Retrieved Memories</h4>';
            details["Memory Search"]["Retrieved Memories"].forEach(memory => {
                html += `<div class="memory-snippet">
                    <div class="memory-text">${formatMessage(memory.text)}</div>
                    <div class="memory-meta">
                        <span>Timestamp: ${memory.timestamp}</span>
                        <span>Similarity: ${(memory.similarity * 100).toFixed(1)}%</span>
                    </div>
                </div>`;
            });
        }
        html += '</div>';
    }
    
    // Memory Synthesis
    if (details["Memory Synthesis"]) {
        html += '<div class="details-section"><h3>Memory Synthesis</h3>';
        html += formatObject(details["Memory Synthesis"]);
        html += '</div>';
    }
    
    // Response Generation
    if (details["Response Generation"]) {
        html += '<div class="details-section"><h3>Response Generation</h3>';
        html += formatObject(details["Response Generation"]);
        html += '</div>';
    }
    
    html += '</div>';
    return html;
}

function formatObject(obj, indent = 0) {
    if (typeof obj !== 'object' || obj === null) {
        return `<span class="value ${typeof obj}">${obj}</span>`;
    }
    
    let html = '';
    if (Array.isArray(obj)) {
        if (obj.length === 0) return '<span class="empty-array">[]</span>';
        html += '<ul class="array">';
        obj.forEach(item => {
            html += `<li>${formatObject(item, indent + 1)}</li>`;
        });
        html += '</ul>';
    } else {
        html += '<div class="object">';
        Object.entries(obj).forEach(([key, value]) => {
            if (key !== "Retrieved Memories") {  // Skip this as it's handled separately
                html += `<div class="property">
                    <span class="key">${key}:</span>
                    ${formatObject(value, indent + 1)}
                </div>`;
            }
        });
        html += '</div>';
    }
    return html;
} 