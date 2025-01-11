document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chat-messages');

    // Handle chat messages
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        messageInput.value = '';
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                addMessageToChat('ai', result.response);
            } else {
                throw new Error(result.detail || 'Failed to get response');
            }
        } catch (error) {
            addMessageToChat('ai', `Error: ${error.message}`);
        }
    }

    function addMessageToChat(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${role}-message`);
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Event listeners for sending messages
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
