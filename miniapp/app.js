(function () {
    'use strict';

    // Centralized application state
    var State = {
        tg: null,
        user: null,
        userId: null,
        theme: 'light',
        channels: [],
        selected: [],
        campaign: null,
        deals: [],
        dealFilter: 'all',
        pollInterval: null,
        lastDealHash: null
    };

    // Configuration
    var API_BASE = '';
    var POLL_DELAY = 30000; // 30 seconds

    // Initialize application
    function init() {
        initTelegram();
        initTheme();
        bindEvents();
        loadChannels();
        hideLoading();
    }

    // Initialize Telegram WebApp
    function initTelegram() {
        if (window.Telegram && window.Telegram.WebApp) {
            State.tg = window.Telegram.WebApp;
            State.tg.ready();
            State.tg.expand();
            if (State.tg.colorScheme) {
                State.theme = State.tg.colorScheme;
            }
        } else {
            State.tg = {
                ready: function () { },
                expand: function () { },
                colorScheme: 'light',
                initDataUnsafe: { user: { id: 0, first_name: 'Demo' } },
                sendData: function (d) { console.log('[sendData]', d); }
            };
        }
        State.user = State.tg.initDataUnsafe?.user || { id: 0, first_name: 'User' };

        // Auth with backend
        if (State.user.id) {
            apiPost('/api/auth', { telegram_id: State.user.id })
                .then(function (res) {
                    if (res.success && res.user) {
                        State.userId = res.user.id;
                        console.log('Authenticated as user:', State.userId);
                    }
                })
                .catch(function (e) {
                    console.log('Auth error:', e);
                });
        }
    }

    // Initialize theme from storage or Telegram
    function initTheme() {
        var saved = localStorage.getItem('adescrow_theme');
        if (saved) {
            State.theme = saved;
        }
        applyTheme(State.theme);
    }

    // Apply theme to document
    function applyTheme(theme) {
        State.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('adescrow_theme', theme);
        var icon = document.getElementById('themeIcon');
        if (icon) {
            icon.textContent = theme === 'dark' ? '☀️' : '🌙';
        }
    }

    // Toggle between light and dark theme
    function toggleTheme() {
        applyTheme(State.theme === 'dark' ? 'light' : 'dark');
    }

    // Bind all event listeners
    function bindEvents() {
        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', toggleTheme);

        // Navigation tabs
        document.querySelectorAll('.nav-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                switchTab(this.dataset.tab);
            });
        });

        // Advertiser flow
        document.getElementById('btnSearch').addEventListener('click', searchChannels);
        document.getElementById('btnProceed').addEventListener('click', function () { showAdvStep(2); });
        document.getElementById('btnBack1').addEventListener('click', function () { showAdvStep(1); });
        document.getElementById('btnSubmitRequest').addEventListener('click', submitRequest);
        document.getElementById('btnNewRequest').addEventListener('click', resetAdvertiserFlow);

        // Channel owner
        document.getElementById('btnRegister').addEventListener('click', registerChannel);

        // Deals filter
        document.querySelectorAll('#dealFilters .filter-chip').forEach(function (chip) {
            chip.addEventListener('click', function () {
                document.querySelectorAll('#dealFilters .filter-chip').forEach(function (c) {
                    c.classList.remove('active');
                });
                this.classList.add('active');
                State.dealFilter = this.dataset.filter;
                renderDeals();
            });
        });
        document.getElementById('btnRefresh').addEventListener('click', loadDeals);
    }

    // Switch between main tabs
    function switchTab(tab) {
        document.querySelectorAll('.nav-btn').forEach(function (b) {
            b.classList.remove('active');
        });
        document.querySelector('[data-tab="' + tab + '"]').classList.add('active');

        document.querySelectorAll('.panel').forEach(function (p) {
            p.classList.remove('active');
        });
        document.getElementById('panel-' + tab).classList.add('active');

        if (tab === 'deals') {
            loadDeals();
            startPolling();
        } else {
            stopPolling();
        }
    }

    // Start polling for deal updates
    function startPolling() {
        if (State.pollInterval) return;
        State.pollInterval = setInterval(refreshDeals, POLL_DELAY);
        console.log('[Polling] Started (every ' + (POLL_DELAY / 1000) + 's)');
    }

    // Stop polling
    function stopPolling() {
        if (State.pollInterval) {
            clearInterval(State.pollInterval);
            State.pollInterval = null;
            console.log('[Polling] Stopped');
        }
    }

    // Refresh deals with smart update (only update UI if data changed)
    function refreshDeals() {
        apiGet('/api/deals')
            .then(function (res) {
                if (res.success && res.deals) {
                    // Create hash to detect changes
                    var newHash = JSON.stringify(res.deals.map(function (d) {
                        return d.id + ':' + d.status;
                    }));

                    if (newHash !== State.lastDealHash) {
                        State.deals = res.deals;
                        State.lastDealHash = newHash;
                        renderDeals();
                        console.log('[Polling] Deals updated');
                    }
                }
            })
            .catch(function (e) {
                console.log('[Polling] Error:', e);
            });
    }

    // Show advertiser step
    function showAdvStep(step) {
        document.querySelectorAll('#panel-advertiser .section').forEach(function (s) {
            s.classList.remove('active');
        });
        document.getElementById('adv-step' + step).classList.add('active');
    }

    // Load channels from backend
    function loadChannels() {
        apiGet('/api/channels')
            .then(function (res) {
                if (res.success && res.channels) {
                    State.channels = res.channels;
                    renderChannels();
                } else {
                    State.channels = [];
                    renderChannels();
                }
            })
            .catch(function (e) {
                console.log('Error loading channels:', e);
                State.channels = [];
                renderChannels();
            });
    }

    // Search and filter channels
    function searchChannels() {
        var category = document.getElementById('filterCategory').value;
        var minSubs = parseInt(document.getElementById('filterSubs').value) || 0;
        var maxPrice = parseInt(document.getElementById('filterPrice').value) || 999;

        // Reload from backend and apply filters
        apiGet('/api/channels')
            .then(function (res) {
                if (res.success && res.channels) {
                    State.channels = res.channels.filter(function (ch) {
                        var catMatch = !category || ch.category === category;
                        var subsMatch = ch.subscribers >= minSubs;
                        var priceMatch = ch.price <= maxPrice;
                        return catMatch && subsMatch && priceMatch;
                    });
                } else {
                    State.channels = [];
                }
                State.selected = [];
                renderChannels();
                updateProceedButton();
            })
            .catch(function (e) {
                console.log('Error searching channels:', e);
                toast('Error loading channels', 'error');
            });
    }

    // Render channel list
    function renderChannels() {
        var container = document.getElementById('channelList');
        var channels = State.channels;

        if (!channels.length) {
            container.innerHTML = '<div class="empty"><div class="empty-icon">📭</div><div class="empty-text">No channels available</div></div>';
            return;
        }

        var html = '';
        channels.forEach(function (ch) {
            var isSelected = State.selected.indexOf(ch.id) !== -1;
            html += '<div class="channel-card' + (isSelected ? ' selected' : '') + '" data-id="' + ch.id + '">' +
                '<div class="channel-check"><svg viewBox="0 0 24 24" fill="none" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg></div>' +
                '<div class="channel-info">' +
                '<div class="channel-name">' + esc(ch.name || ch.handle) + '</div>' +
                '<div class="channel-handle">' + esc(ch.handle) + '</div>' +
                '<div class="channel-meta">' +
                '<span class="meta-tag">' + capitalize(ch.category) + '</span>' +
                '<span class="meta-tag">' + formatNum(ch.subscribers) + ' subs</span>' +
                '<span class="meta-tag">' + formatNum(ch.views || 0) + ' views</span>' +
                '</div>' +
                '</div>' +
                '<div class="channel-price">' +
                '<div class="price-value">' + ch.price + ' TON</div>' +
                '<div class="price-label">per post</div>' +
                '</div>' +
                '</div>';
        });
        container.innerHTML = html;

        // Bind channel selection
        container.querySelectorAll('.channel-card').forEach(function (card) {
            card.addEventListener('click', function () {
                var id = parseInt(this.dataset.id) || this.dataset.id;
                var idx = State.selected.indexOf(id);
                if (idx === -1) {
                    State.selected.push(id);
                    this.classList.add('selected');
                } else {
                    State.selected.splice(idx, 1);
                    this.classList.remove('selected');
                }
                updateProceedButton();
            });
        });
    }

    // Update proceed button visibility
    function updateProceedButton() {
        var btn = document.getElementById('btnProceed');
        var count = document.getElementById('selCount');
        if (State.selected.length > 0) {
            btn.style.display = 'flex';
            count.textContent = State.selected.length;
        } else {
            btn.style.display = 'none';
        }
    }

    // Submit advertising request
    function submitRequest() {
        var title = document.getElementById('campTitle').value.trim();
        var text = document.getElementById('campText').value.trim();
        var budget = parseFloat(document.getElementById('campBudget').value) || 0;
        var schedule = document.getElementById('campSchedule').value;

        if (!title || title.length < 3) {
            toast('Please enter a valid campaign title', 'error');
            return;
        }
        if (!text || text.length < 10) {
            toast('Please enter advertisement text (min 10 characters)', 'error');
            return;
        }
        if (budget < 10) {
            toast('Minimum budget is 10 TON', 'error');
            return;
        }

        setLoading('btnSubmitRequest', true);

        // Create campaign via API
        apiPost('/api/campaign/create', {
            user_id: State.user.id,
            title: title,
            text: text,
            budget: budget
        }).then(function (res) {
            if (res.success && res.campaign) {
                State.campaign = res.campaign;

                // Create deals for selected channels
                var dealPromises = State.selected.map(function (channelId) {
                    var channel = State.channels.find(function (c) { return c.id == channelId; });
                    return apiPost('/api/deal/create', {
                        campaign_id: res.campaign.id,
                        channel_id: channelId,
                        escrow_amount: channel ? channel.price : 0,
                        status: 'pending'
                    });
                });

                return Promise.all(dealPromises);
            }
            throw new Error('Failed to create campaign');
        }).then(function () {
            showConfirmation();
            toast('Request submitted successfully', 'success');
        }).catch(function (e) {
            console.log('Error submitting request:', e);
            toast('Error submitting request', 'error');
        }).finally(function () {
            setLoading('btnSubmitRequest', false);
        });

        // Send to Telegram bot
        sendToBot({
            action: 'submit_request',
            campaign: {
                title: title,
                text: text,
                budget: budget,
                channels: State.selected
            }
        });
    }

    // Show confirmation screen
    function showConfirmation() {
        var total = 0;
        State.selected.forEach(function (id) {
            var ch = State.channels.find(function (c) { return c.id == id; });
            if (ch) total += ch.price;
        });

        var html = '<div class="confirm-block">' +
            '<div class="confirm-label">Campaign Title</div>' +
            '<div class="confirm-value">' + esc(State.campaign.title) + '</div>' +
            '</div>' +
            '<div class="confirm-block">' +
            '<div class="confirm-label">Selected Channels</div>' +
            '<div class="confirm-value">' + State.selected.length + ' channel(s)</div>' +
            '</div>' +
            '<div class="confirm-block">' +
            '<div class="confirm-label">Total Cost</div>' +
            '<div class="confirm-value large">' + total + ' TON</div>' +
            '</div>' +
            '<div class="escrow-banner">' +
            '<div class="escrow-title">Escrow Protection Active</div>' +
            '<div class="escrow-text">Your funds will be held securely in escrow. Payment is only released to channel owners after they publish and verify your advertisement.</div>' +
            '</div>';

        document.getElementById('confirmContent').innerHTML = html;
        showAdvStep(3);
    }

    // Reset advertiser flow
    function resetAdvertiserFlow() {
        State.campaign = null;
        State.selected = [];
        document.getElementById('campTitle').value = '';
        document.getElementById('campText').value = '';
        document.getElementById('campBudget').value = '';
        document.getElementById('filterCategory').value = '';
        document.getElementById('filterSubs').value = '0';
        document.getElementById('filterPrice').value = '999';
        loadChannels();
        updateProceedButton();
        showAdvStep(1);
    }

    // Register channel
    function registerChannel() {
        var handle = document.getElementById('chHandle').value.trim();
        var name = document.getElementById('chName').value.trim();
        var category = document.getElementById('chCategory').value;
        var subs = parseInt(document.getElementById('chSubs').value) || 0;
        var views = parseInt(document.getElementById('chViews').value) || 0;
        var price = parseFloat(document.getElementById('chPrice').value) || 0;

        if (!handle) { toast('Please enter channel username', 'error'); return; }
        if (!handle.startsWith('@')) handle = '@' + handle;
        if (!name) { toast('Please enter channel name', 'error'); return; }
        if (!category) { toast('Please select a category', 'error'); return; }
        if (subs < 100) { toast('Minimum 100 subscribers required', 'error'); return; }
        if (price < 1) { toast('Minimum price is 1 TON', 'error'); return; }

        setLoading('btnRegister', true);

        var data = {
            user_id: State.user.id,
            username: handle,
            name: name,
            category: category,
            subscribers: subs,
            avg_views: views || Math.floor(subs / 5),
            price: price
        };

        apiPost('/api/channels', data).then(function (res) {
            if (res.success) {
                toast('Channel registered successfully', 'success');
                clearChannelForm();
                loadChannels(); // Reload channels list
            } else {
                toast(res.error || 'Failed to register channel', 'error');
            }
        }).catch(function (e) {
            console.log('Error registering channel:', e);
            toast('Error registering channel', 'error');
        }).finally(function () {
            setLoading('btnRegister', false);
        });

        sendToBot({ action: 'register_channel', data: data });
    }

    // Clear channel form
    function clearChannelForm() {
        ['chHandle', 'chName', 'chCategory', 'chSubs', 'chViews', 'chPrice'].forEach(function (id) {
            document.getElementById(id).value = '';
        });
    }

    // Load deals from backend
    function loadDeals() {
        apiGet('/api/deals')
            .then(function (res) {
                if (res.success && res.deals) {
                    State.deals = res.deals;
                } else {
                    State.deals = [];
                }
                renderDeals();
            })
            .catch(function (e) {
                console.log('Error loading deals:', e);
                State.deals = [];
                renderDeals();
            });
    }

    // Render deals with timeline and state machine info
    function renderDeals() {
        var container = document.getElementById('dealList');
        var deals = State.deals;

        if (State.dealFilter !== 'all') {
            deals = deals.filter(function (d) {
                return d.type === State.dealFilter || (State.dealFilter === 'placement' && d.type === 'placement');
            });
        }

        if (!deals.length) {
            container.innerHTML = '<div class="empty"><div class="empty-icon">📋</div><div class="empty-text">No deals found</div></div>';
            return;
        }

        var html = '';
        deals.forEach(function (d) {
            var step = d.step || 1;
            var status = d.status || 'pending';
            var label = d.label || status;
            var isTerminal = d.is_terminal || false;
            var allowedTransitions = d.allowed_transitions || [];

            // Build timeline
            var timeline = '';
            for (var i = 1; i <= 6; i++) {
                var cls = i < step ? 'done' : (i === step ? 'current' : '');
                if (isTerminal && step === 0) cls = 'terminal';
                timeline += '<div class="timeline-step ' + cls + '"></div>';
            }

            // Build action buttons for allowed transitions
            var actions = '';
            if (allowedTransitions.length > 0 && !isTerminal) {
                actions = '<div class="deal-actions">';
                allowedTransitions.forEach(function (nextState) {
                    var btnClass = nextState === 'cancelled' || nextState === 'refunded' ? 'btn-danger' : 'btn-primary';
                    actions += '<button class="btn-sm ' + btnClass + '" onclick="transitionDeal(' + d.id + ', \'' + nextState + '\')">' +
                        nextState.charAt(0).toUpperCase() + nextState.slice(1) + '</button>';
                });
                actions += '</div>';
            }

            html += '<div class="deal-card' + (isTerminal ? ' terminal' : '') + '" data-id="' + d.id + '">' +
                '<div class="deal-timeline">' + timeline + '</div>' +
                '<div class="deal-header">' +
                '<div class="deal-title">' + esc(d.title || 'Deal #' + d.id) + '</div>' +
                '<span class="badge badge-' + status + '">' + esc(label) + '</span>' +
                '</div>' +
                '<div class="deal-meta">' +
                '<span class="deal-channel">' + esc(d.channel || '') + '</span>' +
                '<span class="deal-amount">' + (d.amount || d.escrow_amount || 0) + ' TON</span>' +
                '</div>' +
                actions +
                '</div>';
        });
        container.innerHTML = html;
    }

    // Transition deal to new state
    window.transitionDeal = function (dealId, newState) {
        apiPost('/api/deal/' + dealId + '/transition', {
            state: newState,
            telegram_id: State.user.id
        }).then(function (res) {
            if (res.success) {
                toast('Deal ' + res.transition, 'success');
                loadDeals(); // Reload to reflect changes
            } else {
                toast(res.error || 'Transition failed', 'error');
            }
        }).catch(function (e) {
            console.log('Transition error:', e);
            toast('Error updating deal', 'error');
        });
    };

    // API helper - GET
    function apiGet(url) {
        return fetch(API_BASE + url, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        }).then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        });
    }

    // API helper - POST
    function apiPost(url, data) {
        return fetch(API_BASE + url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        }).then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        });
    }

    // Send data to Telegram bot
    function sendToBot(data) {
        try {
            if (State.tg && State.tg.sendData) {
                State.tg.sendData(JSON.stringify(data));
            }
        } catch (e) {
            console.log('sendToBot error:', e);
        }
    }

    // Set button loading state
    function setLoading(id, loading) {
        var btn = document.getElementById(id);
        if (!btn) return;
        btn.classList.toggle('loading', loading);
        btn.disabled = loading;
    }

    // Show toast notification
    function toast(msg, type) {
        var t = document.getElementById('toast');
        t.textContent = msg;
        t.className = 'toast visible ' + (type || '');
        setTimeout(function () {
            t.classList.remove('visible');
        }, 2500);
    }

    // Hide loading screen
    function hideLoading() {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('app').style.display = 'block';
    }

    // Escape HTML
    function esc(str) {
        var div = document.createElement('div');
        div.textContent = str || '';
        return div.innerHTML;
    }

    // Format number with K/M suffix
    function formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toString();
    }

    // Capitalize first letter
    function capitalize(s) {
        return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
    }

    // Start application
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(init, 60);
    });

    // Expose state for debugging
    window.AdEscrow = State;
})();
