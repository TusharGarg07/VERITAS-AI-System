/**
 * VERITAS AI COMMAND CENTER
 * CORE INTELLIGENCE INTERFACE
 */

const state = {
    trendChart: null,
    lastScanData: null,
    history: [],
    healthProgressCircle: null,
    radius: 70,
    circumference: 2 * Math.PI * 70
};

const API_BASE = '/api/v1';

const elements = {
    // Inputs
    userEmail: document.getElementById('user-email'),
    userName: document.getElementById('user-name'),
    domainSelect: document.getElementById('domain-select'),
    scenarioSelect: document.getElementById('scenario-select'),
    
    simTemp: document.getElementById('sim-temp'),
    simHumid: document.getElementById('sim-humid'),
    simCo2: document.getElementById('sim-co2'),
    simPm25: document.getElementById('sim-pm25'),
    simPm10: document.getElementById('sim-pm10'),
    simTvoc: document.getElementById('sim-tvoc'),

    // Buttons
    runSimBtn: document.getElementById('run-sim'),
    genReportBtn: document.getElementById('gen-report'),
    genPersReportBtn: document.getElementById('gen-pers-report'),
    exportJsonBtn: document.getElementById('export-json'),

    // Displays
    healthProgress: document.getElementById('health-progress'),
    healthScoreVal: document.getElementById('health-score-value'),
    healthStatusLabel: document.getElementById('health-status-label'),
    riskIndex: document.getElementById('risk-index'),
    
    synergyList: document.getElementById('synergy-list'),
    riskList: document.getElementById('risk-list'),
    actionList: document.getElementById('action-list'),
    xaiContainer: document.getElementById('xai-container'),
    
    reportStatus: document.getElementById('report-status'),
    sessionTimestamp: document.getElementById('session-timestamp'),
    notification: document.getElementById('notification')
};

/**
 * INITIALIZATION
 */
function init() {
    initCharts();
    initProgressRing();
    startClock();
    setupEventListeners();
}

function initCharts() {
    const ctx = document.getElementById('liveTrendChart').getContext('2d');
    
    // Cyberpunk Chart Theme
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Rajdhani';

    state.trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CO2 (PPM)',
                    data: [],
                    borderColor: '#00f2ff',
                    backgroundColor: 'rgba(0, 242, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'TEMP (°C)',
                    data: [],
                    borderColor: '#ff0055',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'HUMID (%)',
                    data: [],
                    borderColor: '#00ff9d',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { grid: { display: false } },
                y: { 
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'CO2 Level' }
                },
                y1: {
                    position: 'right',
                    grid: { display: false },
                    title: { display: true, text: 'Temp/Humid' }
                }
            },
            plugins: {
                legend: { position: 'top', align: 'end', labels: { boxWidth: 12, usePointStyle: true } }
            }
        }
    });
}

function initProgressRing() {
    elements.healthProgress.style.strokeDasharray = `${state.circumference} ${state.circumference}`;
    elements.healthProgress.style.strokeDashoffset = state.circumference;
}

function setProgress(percent) {
    const offset = state.circumference - (percent / 100) * state.circumference;
    elements.healthProgress.style.strokeDashoffset = offset;
    
    // Dynamic Color Based on Score
    let color = 'var(--safe)';
    if (percent < 40) color = 'var(--critical)';
    else if (percent < 70) color = 'var(--warning)';
    
    elements.healthProgress.style.stroke = color;
    elements.healthScoreVal.style.color = color;
}

function startClock() {
    setInterval(() => {
        elements.sessionTimestamp.textContent = new Date().toLocaleTimeString();
    }, 1000);
}

/**
 * CORE LOGIC
 */
async function runScan() {
    try {
        setLoading(true);
        notify("INITIATING ENVIRONMENTAL SCAN...");

        // 1. Trigger Simulation
        const simPayload = {
            scenario: elements.scenarioSelect.value,
            duration_minutes: 60,
            sampling_rate_seconds: 300
        };

        const simRes = await fetch(`${API_BASE}/simulation/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(simPayload)
        });

        if (!simRes.ok) throw new Error("Simulation Uplink Failed");
        const simData = await simRes.json();
        state.lastScanData = simData;

        // 2. Fetch Intelligence (Actions & Risks)
        const lastPoint = simData.data_points[simData.data_points.length - 1];
        const intelRes = await fetch(`${API_BASE}/actions/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysis_results: { sensors: lastPoint.sensors } })
        });

        if (!intelRes.ok) throw new Error("Intelligence Processing Failed");
        const intelData = await intelRes.json();

        // 3. Update UI
        updateUI(simData, intelData);
        notify("SCAN COMPLETE. INTELLIGENCE UPDATED.", "safe");

    } catch (err) {
        console.error(err);
        notify(err.message, "critical");
    } finally {
        setLoading(false);
    }
}

function updateUI(simData, intelData) {
    // A. Update Charts
    const points = simData.data_points;
    state.trendChart.data.labels = points.map(p => new Date(p.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
    state.trendChart.data.datasets[0].data = points.map(p => p.sensors.co2);
    state.trendChart.data.datasets[1].data = points.map(p => p.sensors.temperature);
    state.trendChart.data.datasets[2].data = points.map(p => p.sensors.humidity);
    state.trendChart.update();

    // B. Update Health Score
    const score = Math.round((1 - intelData.risk_score) * 100);
    setProgress(score);
    elements.healthScoreVal.textContent = score;
    elements.healthStatusLabel.textContent = score > 80 ? "SYSTEM OPTIMAL" : score > 50 ? "DEGRADED ENVIRONMENT" : "CRITICAL RISK DETECTED";
    elements.healthStatusLabel.style.color = score > 80 ? 'var(--safe)' : score > 50 ? 'var(--warning)' : 'var(--critical)';

    // C. Update Risk Index
    const riskIndex = intelData.risk_score.toFixed(2);
    elements.riskIndex.textContent = riskIndex;
    elements.riskIndex.style.color = riskIndex > 0.7 ? 'var(--critical)' : riskIndex > 0.4 ? 'var(--warning)' : 'var(--safe)';

    // D. Render Actionable Intelligence
    elements.actionList.innerHTML = intelData.actions.map(action => `
        <div class="action-card">
            <h4>${action.priority}: ${action.action}</h4>
            <p>${action.rationale}</p>
        </div>
    `).join('') || '<p class="empty-state">No immediate actions required.</p>';

    // E. Render Primary Risks & Synergies (Using mock logic as backend doesn't return full list yet)
    renderRisks(intelData);
    renderXAI(intelData);
}

function renderRisks(data) {
    // Primary Risks
    const risks = [];
    if (data.risk_score > 0.5) risks.push({ name: "High CO2 Concentration", desc: "Elevated carbon dioxide levels detected." });
    if (data.risk_score > 0.7) risks.push({ name: "VOC Accumulation", desc: "Organic compounds exceed safety thresholds." });

    elements.riskList.innerHTML = risks.map(r => `
        <div class="risk-card">
            <h4>${r.name}</h4>
            <p>${r.desc}</p>
        </div>
    `).join('') || '<p class="empty-state">Environment within safety parameters.</p>';

    // Synergies
    const synergies = [];
    if (data.risk_score > 0.6) synergies.push({ name: "Thermal + CO2 Stress", severity: "warning", desc: "Heat amplifying ventilation issues." });
    
    elements.synergyList.innerHTML = synergies.map(s => `
        <div class="alert-card ${s.severity}">
            <h4>${s.name}</h4>
            <p>${s.desc}</p>
        </div>
    `).join('') || '<p class="empty-state">No synergistic risks detected.</p>';
}

function renderXAI(data) {
    const factors = [
        { name: "CO2 LEVEL", val: 65 },
        { name: "HUMIDITY", val: 20 },
        { name: "TEMPERATURE", val: 15 }
    ];

    elements.xaiContainer.innerHTML = factors.map(f => `
        <div class="xai-item">
            <div class="xai-label">
                <span>${f.name}</span>
                <span>${f.val}%</span>
            </div>
            <div class="xai-bar-bg">
                <div class="xai-bar-fill" style="width: ${f.val}%"></div>
            </div>
        </div>
    `).join('');
}

/**
 * REPORTING
 */
async function generateReport(isPersonalized = false) {
    if (!state.lastScanData) return notify("PLEASE RUN SCAN FIRST", "warning");

    try {
        elements.reportStatus.textContent = "GENERATING INTELLIGENCE...";
        const lastPoint = state.lastScanData.data_points[state.lastScanData.data_points.length - 1];

        let res, data;
        
        if (isPersonalized) {
            const payload = {
                user_name: elements.userName.value || "OPERATOR",
                user_email: elements.userEmail.value,
                domain: elements.domainSelect.value,
                context_type: "Industrial", // Default mapping
                analysis_data: { sensors: lastPoint.sensors }
            };

            if (!payload.user_email) throw new Error("EMAIL REQUIRED FOR PERSONALIZED REPORT");

            res = await fetch(`${API_BASE}/personalized-report/personalized-generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } else {
            res = await fetch(`${API_BASE}/report/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    context_type: "General",
                    analysis_data: { sensors: lastPoint.sensors }
                })
            });
        }

        data = await res.json();
        if (data.file_path) {
            const fileName = data.file_path.split('\\').pop().split('/').pop();
            window.open(`/reports/${fileName}`, '_blank');
            elements.reportStatus.textContent = `REPORT READY: ${data.report_id || 'SUCCESS'}`;
            notify("REPORT GENERATED SUCCESSFULLY", "safe");
        }
    } catch (err) {
        notify(err.message, "critical");
        elements.reportStatus.textContent = "GENERATION FAILED";
    }
}

/**
 * HELPERS
 */
function setupEventListeners() {
    elements.runSimBtn.addEventListener('click', runScan);
    elements.genReportBtn.addEventListener('click', () => generateReport(false));
    elements.genPersReportBtn.addEventListener('click', () => generateReport(true));
    
    elements.exportJsonBtn.addEventListener('click', () => {
        if (!state.lastScanData) return notify("NO DATA TO EXPORT", "warning");
        const blob = new Blob([JSON.stringify(state.lastScanData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `veritas_scan_${Date.now()}.json`;
        a.click();
    });
}

function setLoading(isLoading) {
    elements.runSimBtn.disabled = isLoading;
    elements.runSimBtn.textContent = isLoading ? "SCANNING..." : "INITIATE SCAN";
    elements.runSimBtn.classList.toggle('loading', isLoading);
}

function notify(msg, type = 'accent') {
    elements.notification.textContent = msg;
    elements.notification.className = `notification ${type}`;
    elements.notification.classList.remove('hidden');
    setTimeout(() => elements.notification.classList.add('hidden'), 5000);
}

// Start Command Center
init();
