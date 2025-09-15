let lineChart;
let candlestickChart;
let priceHistory = [];

async function fetchBTCData() {
    try {
        const response = await fetch('/api/btc-data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Fetched data:', data);
        updateDashboard(data);
        updateStatus('🟢 Connected');
    } catch (error) {
        console.error('Error fetching data:', error);
        updateStatus('🔴 Connection Error: ' + error.message);
    }
}

function updateDashboard(data) {
    updatePrice(data.price, data.change24h, data.volume);
    updateIndicators(data);
    updateForecast(data.forecast);
    updateLevels(data.trendLines);
    updateSignals(data.signals);
    updateChart(data.candlesticks, data.trendLines, data.bollinger);
    updateLastUpdate();
}

function updatePrice(price, change, volume) {
    const priceElement = document.getElementById('btc-price');
    const changeElement = document.getElementById('btc-change');
    const volumeElement = document.getElementById('btc-volume');

    priceElement.textContent = `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const safeChange = change || 0;
    const changeText = `${safeChange >= 0 ? '+' : ''}${safeChange.toFixed(2)}%`;
    changeElement.textContent = changeText;
    changeElement.className = `change ${safeChange >= 0 ? 'positive' : 'negative'}`;

    volumeElement.textContent = volume.toLocaleString('en-US');
}

function updateIndicators(data) {
    document.getElementById('rsi-value').textContent = data.rsi.toFixed(2);
    document.getElementById('macd-value').textContent = data.macd.macd.toFixed(4);
    document.getElementById('bb-upper').textContent = `$${data.bollinger.upper.toFixed(2)}`;
    document.getElementById('bb-lower').textContent = `$${data.bollinger.lower.toFixed(2)}`;

    const rsiElement = document.getElementById('rsi-value');
    if (data.rsi < 30) {
        rsiElement.style.color = '#2ecc71';
    } else if (data.rsi > 70) {
        rsiElement.style.color = '#e74c3c';
    } else {
        rsiElement.style.color = '#f39c12';
    }
}

function updateForecast(forecast) {
    if (!forecast) return;

    document.getElementById('forecast-short').textContent = `$${forecast.shortTerm.toFixed(2)}`;
    document.getElementById('forecast-medium').textContent = `$${forecast.mediumTerm.toFixed(2)}`;
    document.getElementById('forecast-confidence').textContent = `${forecast.confidence.toFixed(1)}%`;

    const trendElement = document.getElementById('forecast-trend');
    trendElement.textContent = forecast.direction.charAt(0).toUpperCase() + forecast.direction.slice(1);
    trendElement.className = `forecast-value trend-${forecast.direction}`;
}

function updateLevels(trendLines) {
    if (!trendLines) return;

    // Update resistance levels
    const resistanceContainer = document.getElementById('resistance-levels');
    if (trendLines.resistance && trendLines.resistance.length > 0) {
        resistanceContainer.innerHTML = trendLines.resistance.map(level => `
            <div class="level-item resistance">
                <span class="level-price">$${level.price.toFixed(2)}</span>
                <span class="level-strength">Strength: ${level.strength}</span>
            </div>
        `).join('');
    } else {
        resistanceContainer.innerHTML = '<div class="no-levels">No resistance levels detected</div>';
    }

    // Update support levels
    const supportContainer = document.getElementById('support-levels');
    if (trendLines.support && trendLines.support.length > 0) {
        supportContainer.innerHTML = trendLines.support.map(level => `
            <div class="level-item support">
                <span class="level-price">$${level.price.toFixed(2)}</span>
                <span class="level-strength">Strength: ${level.strength}</span>
            </div>
        `).join('');
    } else {
        supportContainer.innerHTML = '<div class="no-levels">No support levels detected</div>';
    }
}

function updateSignals(signals) {
    const signalsList = document.getElementById('signals-list');

    if (!signals || signals.length === 0) {
        signalsList.innerHTML = '<div class="no-signals">No signals detected</div>';
        return;
    }

    signalsList.innerHTML = signals.map(signal => `
        <div class="signal ${signal.type.toLowerCase()}">
            <div class="signal-header">
                <span class="signal-type">${signal.type}</span>
                <span class="signal-strength">${signal.strength}</span>
            </div>
            <div class="signal-indicator">${signal.indicator}</div>
            <div class="signal-message">${signal.message}</div>
            <div class="signal-time">${new Date(signal.timestamp).toLocaleTimeString()}</div>
        </div>
    `).join('');
}

function updateChart(candlesticks, trendLines, bollinger) {
    if (!candlesticks || candlesticks.length === 0) return;

    const labels = candlesticks.map(candle =>
        new Date(candle.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    );
    const prices = candlesticks.map(candle => candle.close);

    // Update Line Chart with Bollinger Bands
    const datasets = [{
        label: 'BTC Price',
        data: prices,
        borderColor: '#f39c12',
        backgroundColor: 'rgba(243, 156, 18, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.4
    }];

    // Add Bollinger Bands if available
    if (bollinger && bollinger.upper && bollinger.lower) {
        const upperBand = new Array(prices.length).fill(bollinger.upper);
        const lowerBand = new Array(prices.length).fill(bollinger.lower);
        const middleBand = new Array(prices.length).fill(bollinger.middle);

        datasets.push({
            label: 'BB Upper',
            data: upperBand,
            borderColor: 'rgba(231, 76, 60, 0.6)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        });

        datasets.push({
            label: 'BB Middle',
            data: middleBand,
            borderColor: 'rgba(149, 165, 166, 0.6)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [3, 3],
            fill: false,
            pointRadius: 0
        });

        datasets.push({
            label: 'BB Lower',
            data: lowerBand,
            borderColor: 'rgba(46, 204, 113, 0.6)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        });
    }

    if (!lineChart) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        lineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: 'white',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    } else {
        lineChart.data.labels = labels;
        lineChart.data.datasets = datasets;
        lineChart.update('none');
    }

    // Update Candlestick Chart
    const candlestickData = candlesticks.map(candle => ({
        x: candle.time,
        o: candle.open,
        h: candle.high,
        l: candle.low,
        c: candle.close
    }));

    if (!candlestickChart) {
        const ctx2 = document.getElementById('candlestickChart').getContext('2d');
        candlestickChart = new Chart(ctx2, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: 'BTC/USDT',
                    data: candlestickData,
                    borderColor: {
                        up: '#2ecc71',
                        down: '#e74c3c',
                        unchanged: '#95a5a6'
                    },
                    backgroundColor: {
                        up: 'rgba(46, 204, 113, 0.8)',
                        down: 'rgba(231, 76, 60, 0.8)',
                        unchanged: 'rgba(149, 165, 166, 0.8)'
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            displayFormats: {
                                minute: 'HH:mm'
                            }
                        },
                        ticks: {
                            color: 'white',
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: 'white',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    } else {
        candlestickChart.data.datasets[0].data = candlestickData;
        candlestickChart.update('none');
    }
}

function updateStatus(status) {
    document.getElementById('status').textContent = status;
}

function updateLastUpdate() {
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
}

setInterval(fetchBTCData, 10000);

fetchBTCData();