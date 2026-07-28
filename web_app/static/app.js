// Real-time routing checker (Disease Routing Module - DRM)
function updateRoutingStatus() {
    const form = document.getElementById("intake-form");
    const age = form.age.value;
    const gender = form.gender.value;
    const smoking = form.smoking.value;
    const bmi = form.bmi.value;
    const wheezing = form.wheezing.value;
    const allergies = form.allergies.value;
    const fev1 = form.fev1.value;
    const fvc = form.fvc.value;
    const fvc_percent = form.fvc_percent.value;
    const fever = form.fever.value;
    const cough = form.cough.value;
    const chest_pain = form.chest_pain.value;
    const wbc_count = form.wbc_count.value;
    const spo2 = form.spo2.value;
    const resp_symptom = form.resp_symptom.value;
    const resp_rate = form.resp_rate.value;
    const dyspnea = form.dyspnea.value;
    const heart_rate = form.heart_rate.value;

    // Define indicators
    const indicators = [
        {
            name: "Asthma",
            colorClass: "asthma",
            active: !!(age && gender && wheezing && allergies && smoking && fev1),
            missing: getMissingFields({ age, gender, wheezing, allergies, smoking, fev1 }, {
                age: "Age", gender: "Gender", wheezing: "Wheezing", allergies: "Allergies", smoking: "Smoking", fev1: "FEV1"
            })
        },
        {
            name: "IPF",
            colorClass: "ipf",
            active: !!(fvc && fvc_percent && age && gender && smoking),
            missing: getMissingFields({ fvc, fvc_percent, age, gender, smoking }, {
                fvc: "FVC (mL)", fvc_percent: "FVC % predicted", age: "Age", gender: "Gender", smoking: "Smoking"
            })
        },
        {
            name: "Pneumonia",
            colorClass: "pneumonia",
            active: !!(fever && cough && chest_pain && spo2 && resp_symptom),
            missing: getMissingFields({ fever, cough, chest_pain, spo2, resp_symptom }, {
                fever: "Fever", cough: "Cough", chest_pain: "Chest Pain", spo2: "SpO2", resp_symptom: "Shortness of breath"
            })
        },
        {
            name: "COPD",
            colorClass: "copd",
            active: !!(age && gender && smoking && bmi && fev1 && spo2 && resp_rate && dyspnea && heart_rate),
            missing: getMissingFields({ age, gender, smoking, bmi, fev1, spo2, resp_rate, dyspnea, heart_rate }, {
                age: "Age", gender: "Gender", smoking: "Smoking", bmi: "BMI", fev1: "FEV1", spo2: "SpO2", resp_rate: "Respiration Rate", dyspnea: "Dyspnea", heart_rate: "Heart Rate"
            })
        }
    ];

    // Render indicators
    const container = document.getElementById("routing-indicators");
    container.innerHTML = "";

    let anyActive = false;
    indicators.forEach(ind => {
        if (ind.active) anyActive = true;
        
        const row = document.createElement("div");
        row.className = `routing-indicator-row ${ind.active ? 'active' : ''}`;
        
        row.innerHTML = `
            <div class="disease-info">
                <span class="badge-dot ${ind.colorClass}"></span>
                <h4>${ind.name} Dispatcher</h4>
            </div>
            <div class="routing-status ${ind.active ? 'active' : 'inactive'}">
                ${ind.active ? 'Active ✅' : 'Inactive ✖'}
            </div>
        `;
        
        if (!ind.active && ind.missing.length > 0) {
            const missingSpan = document.createElement("span");
            missingSpan.className = "missing-params";
            missingSpan.innerText = `Requires: ${ind.missing.join(", ")}`;
            row.appendChild(missingSpan);
        }
        
        container.appendChild(row);
    });

    // Enable/disable central prediction button
    const predictBtn = document.getElementById("btn-predict");
    predictBtn.disabled = !anyActive;
}

function getMissingFields(fieldsObj, namesMap) {
    let missing = [];
    for (let key in fieldsObj) {
        if (!fieldsObj[key]) {
            missing.append(namesMap[key]); // wait! append doesn't exist on standard array! It's push! Let's correct.
        }
    }
    return missing;
}

// Correcting the getMissingFields implementation to use push
function getMissingFields(fieldsObj, namesMap) {
    let missing = [];
    for (let key in fieldsObj) {
        if (!fieldsObj[key]) {
            missing.push(namesMap[key]);
        }
    }
    return missing;
}

// Post form data to route and score predictions
async function runPredictions() {
    const form = document.getElementById("intake-form");
    const payload = {
        age: form.age.value,
        gender: form.gender.value,
        smoking: form.smoking.value,
        bmi: form.bmi.value,
        wheezing: form.wheezing.value,
        allergies: form.allergies.value,
        fev1: form.fev1.value,
        fvc: form.fvc.value,
        fvc_percent: form.fvc_percent.value,
        fever: form.fever.value,
        cough: form.cough.value,
        chest_pain: form.chest_pain.value,
        wbc_count: form.wbc_count.value,
        spo2: form.spo2.value,
        resp_symptom: form.resp_symptom.value,
        resp_rate: form.resp_rate.value,
        dyspnea: form.dyspnea.value,
        heart_rate: form.heart_rate.value
    };

    try {
        const predictBtn = document.getElementById("btn-predict");
        predictBtn.innerText = "Analyzing Patient Case...";
        predictBtn.disabled = true;

        const response = await fetch("/route_and_predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await response.json();

        predictBtn.innerText = "Analyze Patient Case";
        predictBtn.disabled = false;

        renderResults(result.predictions);

    } catch (err) {
        console.error("Prediction failed:", err);
        alert("An error occurred during diagnostics. Check backend API logs.");
        const predictBtn = document.getElementById("btn-predict");
        predictBtn.innerText = "Analyze Patient Case";
        predictBtn.disabled = false;
    }
}

// Render dynamic results gauges and cards
function renderResults(predictions) {
    const noPredMessage = document.getElementById("no-active-predictions");
    const resultsGrid = document.getElementById("prediction-results");
    
    // Clear grid
    resultsGrid.innerHTML = "";

    const activeKeys = Object.keys(predictions);

    if (activeKeys.length === 0) {
        noPredMessage.style.display = "flex";
        resultsGrid.classList.add("hide");
        return;
    }

    noPredMessage.style.display = "none";
    resultsGrid.classList.remove("hide");

    activeKeys.forEach(diseaseName => {
        const pred = predictions[diseaseName];
        if (pred.status === "Error") {
            const errorCard = document.createElement("div");
            errorCard.className = `result-card-item ${diseaseName.toLowerCase()}`;
            errorCard.innerHTML = `
                <div class="result-header">
                    <h3 class="${diseaseName.toLowerCase()}">${diseaseName} Diagnostics</h3>
                    <span class="confidence-val">Error</span>
                </div>
                <div class="result-display-area">
                    <p style="color: #ef4444; font-size: 0.85rem;">Failed: ${pred.message}</p>
                </div>
            `;
            resultsGrid.appendChild(errorCard);
            return;
        }

        const classGlowColor = diseaseName.toLowerCase();
        const confPercent = (pred.confidence * 100).toFixed(1);
        
        const card = document.createElement("div");
        card.className = `result-card-item ${classGlowColor}`;
        
        // Formulate probability breakdown
        let breakdownHTML = "";
        if (pred.stages && pred.probability) {
            breakdownHTML = `<div class="multiclass-breakdown">`;
            pred.stages.forEach((stageName, idx) => {
                const isHighest = stageName === pred.class;
                const pVal = (pred.probability[idx] * 100).toFixed(1);
                breakdownHTML += `
                    <div class="breakdown-row ${isHighest ? 'highest' : ''}">
                        <span>${stageName}</span>
                        <span>${pVal}%</span>
                    </div>
                `;
            });
            breakdownHTML += `</div>`;
        } else if (pred.probability) {
            // Binary case breakdown
            const isPos = pred.class.includes("Detected") || pred.class.includes("1");
            const posVal = (pred.probability[1] * 100).toFixed(1);
            const negVal = (pred.probability[0] * 100).toFixed(1);
            breakdownHTML = `
                <div class="multiclass-breakdown">
                    <div class="breakdown-row ${!isPos ? 'highest' : ''}">
                        <span>Negative Risk</span>
                        <span>${negVal}%</span>
                    </div>
                    <div class="breakdown-row ${isPos ? 'highest' : ''}">
                        <span>Positive Risk</span>
                        <span>${posVal}%</span>
                    </div>
                </div>
            `;
        }

        card.innerHTML = `
            <div class="result-header">
                <h3 class="${classGlowColor}">${diseaseName} Diagnostics</h3>
                <span class="confidence-val">Confidence: ${confPercent}%</span>
            </div>
            <div class="result-display-area">
                <div class="class-output">${pred.class}</div>
                <div class="prob-meter-container">
                    <div class="prob-meter-fill ${classGlowColor}" style="width: ${confPercent}%"></div>
                </div>
                ${breakdownHTML}
            </div>
        `;
        
        resultsGrid.appendChild(card);
    });
}

// Initial draw on page load
window.onload = function() {
    updateRoutingStatus();
};
