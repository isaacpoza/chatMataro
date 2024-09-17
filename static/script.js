// Mostrar/Ocultar el chat
function toggleChat() {
    const chatContainer = document.getElementById("chat-container");
    if (chatContainer.classList.contains("hidden")) {
        chatContainer.classList.remove("hidden");
    } else {
        chatContainer.classList.add("hidden");
    }
}

// Minimizar/Maximizar el chat
function minimizeChat() {
    const chatContainer = document.getElementById("chat-container");
    const chatLog = document.getElementById("chat-log");
    const chatInputContainer = document.getElementById("chat-input-container");

    if (chatContainer.classList.contains("minimized")) {
        chatContainer.classList.remove("minimized");
        chatLog.style.display = "block";
        chatInputContainer.style.display = "flex";
    } else {
        chatContainer.classList.add("minimized");
        chatLog.style.display = "none";
        chatInputContainer.style.display = "none";
    }
}

// Función para enviar el mensaje
document.getElementById("send-btn").addEventListener("click", function() {
    var userInput = document.getElementById("user-input").value;
    document.getElementById("user-input").value = "";

    var chatLog = document.getElementById("chat-log");
    chatLog.innerHTML += "<div><strong>Tú:</strong> " + userInput + "</div>";

    fetch("/get_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        chatLog.innerHTML += "<div><strong>Bot:</strong> " + data.response + "</div>";
        chatLog.scrollTop = chatLog.scrollHeight;
    });
});

// Hacer que el chat se pueda arrastrar
dragElement(document.getElementById("chat-container"));

function dragElement(elmnt) {
    var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    document.getElementById("chat-header").onmousedown = dragMouseDown;

    function dragMouseDown(e) {
        e = e || window.event;
        e.preventDefault();
        pos3 = e.clientX;
        pos4 = e.clientY;
        document.onmouseup = closeDragElement;
        document.onmousemove = elementDrag;
        elmnt.classList.add("dragging");
    }

    function elementDrag(e) {
        e = e || window.event;
        e.preventDefault();
        pos1 = pos3 - e.clientX;
        pos2 = pos4 - e.clientY;
        pos3 = e.clientX;
        pos4 = e.clientY;
        elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
        elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
    }

    function closeDragElement() {
        document.onmouseup = null;
        document.onmousemove = null;
        elmnt.classList.remove("dragging");
    }
}
