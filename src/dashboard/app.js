const transactionFeed = document.getElementById('transaction-feed');
const metricTotal = document.getElementById('metric-total');
const metricBlocked = document.getElementById('metric-blocked');
const geminiAnalysis = document.getElementById('gemini-analysis');
const featureBars = document.getElementById('feature-bars');
const emptyFeedMsg = document.getElementById('feed-empty');

let totalCount = 0;
let blockedCount = 0;
let transactions = new Map(); // Store full txn data by ID

// Format currency
const formatter = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 0
});

// Setup SSE Stream
function connectStream() {
    console.log("Connecting to live feed...");
    const eventSource = new EventSource('/api/stream');

    eventSource.addEventListener('transaction', (event) => {
        if (emptyFeedMsg) emptyFeedMsg.style.display = 'none';
        
        const data = JSON.parse(event.data);
        transactions.set(data.transaction_id, data);
        
        // Update Metrics
        totalCount++;
        metricTotal.innerText = totalCount.toLocaleString();
        
        if (data.is_flagged) {
            blockedCount++;
            metricBlocked.innerText = blockedCount.toLocaleString();
        }

        renderTransactionRow(data);
    });

    eventSource.addEventListener('complete', () => {
        console.log("Stream complete.");
        eventSource.close();
    });

    eventSource.onerror = (error) => {
        console.error("EventSource failed:", error);
        eventSource.close();
        setTimeout(connectStream, 5000); // Reconnect logic
    };
}

function renderTransactionRow(txn) {
    const row = document.createElement('div');
    row.className = `txn-row ${txn.is_flagged ? 'flagged' : ''}`;
    row.onclick = () => selectTransaction(row, txn.transaction_id);

    const scoreTag = txn.is_flagged 
        ? `<div class="tag risk">Risk: ${(txn.fraud_score * 100).toFixed(1)}%</div>` 
        : `<div class="tag safe">Safe: ${(txn.fraud_score * 100).toFixed(1)}%</div>`;

    row.innerHTML = `
        <div class="txn-id">${txn.transaction_id}</div>
        <div class="txn-amount">${formatter.format(txn.amount)}</div>
        <div class="txn-city">${txn.customer_city || 'Unknown'}</div>
        <div class="txn-method" style="text-transform: capitalize;">${txn.payment_method}</div>
        ${scoreTag}
    `;

    transactionFeed.insertBefore(row, transactionFeed.firstChild);

    // Keep only last 100 in DOM to prevent lag
    if (transactionFeed.children.length > 100) {
        transactionFeed.removeChild(transactionFeed.lastChild);
    }
}

function selectTransaction(element, id) {
    // UI Selection State
    document.querySelectorAll('.txn-row').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');

    const txn = transactions.get(id);
    if (!txn) return;

    // Update Gemini Analysis Box
    if (txn.is_flagged) {
        geminiAnalysis.innerHTML = `
            <p><strong>🚨 Transaction Blocked (Score: ${(txn.fraud_score * 100).toFixed(1)}%)</strong></p>
            <p>${txn.explanation || "No explanation provided by LLM."}</p>
        `;
        
        // Render Feature Importance Bars
        renderFeatures(txn.top_features);
    } else {
        geminiAnalysis.innerHTML = `
            <p><strong>✅ Transaction Approved (Score: ${(txn.fraud_score * 100).toFixed(1)}%)</strong></p>
            <p>This transaction matches normal behavioral patterns for the merchant and cardholder. No anomalies detected.</p>
        `;
        featureBars.innerHTML = '<div class="empty-analysis">Normal transaction pattern.</div>';
    }
}

function renderFeatures(featuresArray) {
    if (!featuresArray || featuresArray.length === 0) {
        featureBars.innerHTML = '<div class="empty-analysis">No feature data available.</div>';
        return;
    }

    featureBars.innerHTML = '';
    
    // Parse "feature=value" strings and find max value for relative bar widths
    let maxAbsValue = 0.01;
    const parsedFeatures = featuresArray.map(f => {
        let name = f, value = 0;
        if (f.includes('=')) {
            const parts = f.split('=');
            name = parts[0];
            value = parseFloat(parts[1]) || 0;
        }
        maxAbsValue = Math.max(maxAbsValue, Math.abs(value));
        return { name, value };
    });

    parsedFeatures.forEach(f => {
        const percent = Math.min((Math.abs(f.value) / maxAbsValue) * 100, 100);
        
        const container = document.createElement('div');
        container.className = 'feature-bar-container';
        container.innerHTML = `
            <div class="feature-label">
                <span>${f.name}</span>
                <span>${f.value.toFixed(2)}</span>
            </div>
            <div class="feature-track">
                <div class="feature-fill" style="width: 0%"></div>
            </div>
        `;
        featureBars.appendChild(container);

        // Trigger animation after append
        setTimeout(() => {
            container.querySelector('.feature-fill').style.width = `${percent}%`;
        }, 50);
    });
}

// Start app
connectStream();
