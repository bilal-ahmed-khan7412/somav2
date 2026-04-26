const startBtn = document.getElementById('start-btn');
const terminalContent = document.getElementById('terminal-content');
const negotiationFeed = document.getElementById('negotiation-feed');
const pulseDot = document.querySelector('.pulse-dot');

// Agent mapping for labels
const agentMapping = {
    'agent_0': 'SECOPS',
    'agent_1': 'NETOPS',
    'agent_2': 'SYSADMIN'
};

const agentColorClass = {
    'agent_0': 'COMMAND',
    'agent_1': 'FIELD',
    'agent_2': 'SUPPORT'
};

function formatTime() {
    const now = new Date();
    return now.toISOString().substring(11, 23);
}

function appendTerminalLog(message, level="INFO", isLLM=false) {
    const div = document.createElement('div');
    div.className = 'log-line';
    
    let content = `<span class="log-time">[${formatTime()}]</span> `;
    if (isLLM) {
        content += `<span class="log-llm">${message}</span>`;
    } else {
        content += `<span class="log-level-${level}">[${level}]</span> <span style="color:#d1d5db">${message}</span>`;
    }
    
    div.innerHTML = content;
    terminalContent.appendChild(div);
    
    // Smooth scroll to bottom
    setTimeout(() => {
        terminalContent.scrollTop = terminalContent.scrollHeight;
    }, 10);
}

function updateServerState(serverName, state, statusText) {
    const card = document.getElementById(serverName);
    if (!card) return;
    
    card.className = `server-card state-${state}`;
    const statusEl = card.querySelector('.server-status');
    if (statusEl) statusEl.textContent = statusText;
}

function setServerLock(serverName, agentId) {
    const lockEl = document.getElementById(`lock-${serverName}`);
    if (lockEl) {
        if (agentId) {
            lockEl.textContent = `🔒 ${agentMapping[agentId] || agentId}`;
            lockEl.className = 'lock-indicator visible';
        } else {
            lockEl.className = 'lock-indicator';
        }
    }
}

function setAgentTask(agentId, taskText, isBusy = true) {
    const card = document.getElementById(agentId);
    if (!card) return;
    
    if (isBusy) {
        card.classList.add('busy');
        card.querySelector('.load-badge').textContent = 'Busy';
    } else {
        card.classList.remove('busy');
        card.querySelector('.load-badge').textContent = 'Idle';
    }
    card.querySelector('.agent-task').textContent = taskText || (isBusy ? 'Processing...' : 'Task Completed - Idle');
}

function appendNegotiation(msgType, sender, target, text) {
    const div = document.createElement('div');
    
    // Map MsgType Enum text
    let typeClass = 'PROPOSAL';
    if (msgType.includes('ACCEPT') || msgType.includes('COMPLETE')) typeClass = 'ACCEPT';
    if (msgType.includes('REJECT')) typeClass = 'REJECT';
    if (msgType.includes('DELEGATION')) typeClass = 'HANDSHAKE';
    
    div.className = `neg-message neg-msg-${typeClass}`;
    
    let senderName = agentMapping[sender] || sender;
    let targetName = agentMapping[target] || target;
    if (target === 'swarm_pool') targetName = '🔄 SWARM POOL';
    if (sender === 'swarm_pool') senderName = '🔄 SWARM POOL';
    
    let dirArrow = '→';
    if (typeClass === 'ACCEPT') dirArrow = '✓';
    if (typeClass === 'REJECT') dirArrow = '✗';
    if (typeClass === 'HANDSHAKE') dirArrow = '⚡';

    div.innerHTML = `
        <div class="neg-participants">
            <span style="color:var(--agent-${agentColorClass[sender]?.toLowerCase() || 'field'})">${senderName}</span> 
            ${dirArrow} 
            <span style="color:var(--agent-${agentColorClass[target]?.toLowerCase() || 'field'})">${targetName}</span>
        </div>
        <div class="neg-text">${text}</div>
    `;
    
    const hint = negotiationFeed.querySelector('.log-hint');
    if (hint) hint.remove();
    
    negotiationFeed.appendChild(div);
    setTimeout(() => {
        negotiationFeed.scrollTop = negotiationFeed.scrollHeight;
    }, 10);
}

// Map A2A message types to UI actions
function handleBusMessage(msg) {
    const msgType = msg.msg_type.split('.').pop(); // e.g. MsgType.RESOURCE_CLAIM -> RESOURCE_CLAIM
    const payload = msg.payload || {};
    
    if (msgType === 'RESOURCE_CLAIM') {
        setServerLock(payload.unit_id, msg.sender);
        appendTerminalLog(`${msg.sender} locked ${payload.unit_id}`, "DEBUG");
    } 
    else if (msgType === 'RESOURCE_RELEASE') {
        setServerLock(payload.unit_id, null);
        appendTerminalLog(`${msg.sender} released ${payload.unit_id}`, "DEBUG");
    }
    else if (msgType === 'STEP_PROPOSAL' || msgType === 'DELEGATION_PROPOSAL') {
        appendNegotiation(msgType, msg.sender, msg.recipient, payload.step || 'Requested execution of step');
    }
    else if (msgType === 'STEP_ACCEPT' || msgType === 'DELEGATION_COMPLETE') {
        appendNegotiation(msgType, msg.sender, msg.recipient, payload.result || 'Accepted and executed step');
    }
    else if (msgType === 'STEP_REJECT') {
        appendNegotiation('REJECT', msg.sender, msg.recipient, payload.reason || 'Rejected proposal');
    }
    else if (msgType === 'TASK_RESULT') {
        appendTerminalLog(`[A2A] Task resolved by ${msg.sender}: ${payload.task}`, "INFO");
        setAgentTask(msg.sender, '', false);
    }
}

// Connect SSE stream
const evtSource = new EventSource('/stream');

evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'internal_log') {
        appendTerminalLog(data.msg, data.level);
    }
    else if (data.type === 'llm_log') {
        appendTerminalLog(data.msg, 'INFO', true);
    }
    else if (data.type === 'a2a_log') {
        handleBusMessage(data.msg);
    }
    else if (data.type === 'scenario_cmd') {
        // Custom scenario commands sent from python
        if (data.action === 'set_server_state') {
            updateServerState(data.server, data.state, data.statusText);
        }
        else if (data.action === 'set_agent_task') {
            setAgentTask(data.agent, data.task, data.isBusy !== false);
        }
        else if (data.action === 'simulation_end') {
            startBtn.disabled = false;
            startBtn.textContent = 'RESTART SIMULATION';
            pulseDot.style.backgroundColor = 'var(--color-success)';
            appendTerminalLog("SIMULATION COMPLETED.", "INFO");
        }
    }
};

evtSource.onerror = function() {
    console.error("EventSource failed.");
};


// Start Button
startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    startBtn.textContent = 'SIMULATION RUNNING...';
    pulseDot.style.backgroundColor = 'var(--color-danger)';
    
    terminalContent.innerHTML = '';
    negotiationFeed.innerHTML = '<div class="log-hint">Awaiting inter-agent proposals...</div>';
    
    appendTerminalLog("Initiating Sentinel Simulation via POST /simulate...", "INFO");
    
    // Reset UI
    ['PROXY_NODE', 'APP_NODE_1', 'APP_NODE_2', 'DB_PRIMARY', 'DB_REPLICA'].forEach(s => {
        updateServerState(s, 'healthy', 'Healthy');
        setServerLock(s, null);
    });
    
    try {
        await fetch('/simulate', { method: 'POST' });
    } catch (e) {
        appendTerminalLog("Failed to start simulation: " + e.message, "ERROR");
        startBtn.disabled = false;
    }
});
