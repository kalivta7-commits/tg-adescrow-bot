import { createClient } from "https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm";

(function () {
    'use strict';

    const SUPABASE_URL = 'https://ljplgcuzvsylgrwkwmoq.supabase.co';
    const SUPABASE_ANON_KEY = 'sb_publishable_QzCFORKqw8d5Y2w6MA4KRw_AFJb4kYx';
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

    async function ensureAuthenticated() {
        const { data: { session } } = await supabase.auth.getSession();

        if (session) {
            return session;
        }

        // Temporary dev login (REMOVE later in production)
        const { data, error } = await supabase.auth.signInAnonymously();

        if (error) {
            console.error("Supabase anonymous login failed:", error);
            return null;
        }

        return data.session;
    }

    window.addEventListener('error', function (event) {
        console.error('[GlobalError]', event && event.error ? event.error : event);
        toast('Something went wrong. Please try again.', 'error');
        if (event && typeof event.preventDefault === 'function') {
            event.preventDefault();
        }
        return true;
    });

    window.addEventListener('unhandledrejection', function (event) {
        console.error('[UnhandledRejection]', event && event.reason ? event.reason : event);
        toast('Request failed. Please retry.', 'error');
        if (event && typeof event.preventDefault === 'function') {
            event.preventDefault();
        }
    });

    function safeObj(value) {
        return value && typeof value === 'object' ? value : {};
    }

    function safeArray(value) {
        return Array.isArray(value) ? value : [];
    }

    function toText(value, fallback) {
        if (value === null || value === undefined) {
            return fallback || '';
        }
        return String(value);
    }


    function toNumber(value, fallback) {
        var n = Number(value);
        return Number.isFinite(n) ? n : (fallback || 0);
    }

    function esc(str) {
        var div = document.createElement('div');
        div.textContent = toText(str, '');
        return div.innerHTML;
    }

    function formatNum(n) {
        var num = toNumber(n, 0);
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return toText(num);
    }

    function capitalize(s) {
        var text = toText(s, '');
        return text ? text.charAt(0).toUpperCase() + text.slice(1) : '';
    }

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
                    var user = safeObj((res && res.user) || (res && res.data && res.data.user));
                    if (res && res.success && user.id) {
                        State.userId = user.id;
                        console.log('Authenticated as user:', State.userId);
                    } else {
                        throw new Error((res && res.error) || 'Auth response missing user id');
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
                renderDeals(State.deals);
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

        if (tab === 'leaderboard') {
            loadLeaderboard();
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
        var telegramId = State.user && State.user.id ? State.user.id : 0;
        var channelId = State.user && (State.user.channelId || State.user.channel_id || null);
        if (!telegramId) return;

        var url = '/api/deals?telegram_id=' + encodeURIComponent(telegramId);
        if (channelId) {
            url += '&user_channel_uid=' + encodeURIComponent(channelId);
        }

        apiGet(url)
            .then(function (res) {
                var deals = safeArray(res && res.data);

                // Create hash to detect changes
                var newHash = JSON.stringify(deals.map(function (d) {
                    var deal = safeObj(d);
                    return toText(deal.id, '0') + ':' + toText(deal.status, 'pending');
                }));

                if (newHash !== State.lastDealHash) {
                    State.deals = deals;
                    State.lastDealHash = newHash;
                    renderDeals(State.deals);
                    console.log('[Polling] Deals updated');
                }
            })
            .catch(function (e) {
                console.error('Deals fetch error', e);
                State.deals = [];
                State.lastDealHash = null;
                renderDeals(State.deals);
            });
    }

    function submitDeal(selectedCampaign, selectedChannel, amountValue) {
        var payload = {
            telegram_id: State.user.id,
            campaign_id: selectedCampaign.id,
            channel_id: selectedChannel.id,
            amount: Number(amountValue)
        };

        return apiPost('/api/deal/create', payload)
            .then(function (res) {
                if (res && res.data) {
                    console.log('Deal created:', res.data);
                    return res.data;
                }
                throw new Error((res && res.error) || 'Failed to create deal');
            })
            .catch(function (err) {
                console.error('Create deal', err);
                throw err;
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
                State.channels = res && res.success === true && Array.isArray(res.data) ? res.data : [];
                renderChannels();
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
                if (res && res.success === true && Array.isArray(res.data)) {
                    State.channels = res.data.filter(function (ch) {
                        var channel = safeObj(ch);
                        var catMatch = !category || toText(channel.category).toLowerCase() === toText(category).toLowerCase();
                        var subsMatch = toNumber(channel.subscribers, 0) >= minSubs;
                        var priceMatch = toNumber(channel.price, 0) <= maxPrice;
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
        if (!container) return;
        var channels = safeArray(State.channels);

        if (!channels.length) {
            container.innerHTML = '<div class="empty"><div class="empty-icon">📭</div><div class="empty-text">No channels available</div></div>';
            return;
        }

        var html = '';
        channels.forEach(function (ch, index) {
            var channel = safeObj(ch);
            var channelId = toText(channel.id, '').trim();
            if (!channelId) {
                return;
            }
            var channelName = toText(channel.name || channel.username || 'Unknown channel');
            var channelUser = toText(channel.username || '');
            var channelCategory = capitalize(toText(channel.category || 'general'));
            var channelSubs = formatNum(toNumber(channel.subscribers, 0));
            var channelViews = formatNum(toNumber(channel.avg_views, 0));
            var channelPrice = toNumber(channel.price, 0);
            var channelPublicLink = toText(channel.public_link || '');
            var hasSuccessRate = channel.success_rate !== undefined && channel.success_rate !== null;
            var isSelected = State.selected.indexOf(channelId) !== -1;
            html += '<div class="channel-card' + (isSelected ? ' selected' : '') + '" data-id="' + esc(channelId) + '">' +
                '<div class="channel-check"><svg viewBox="0 0 24 24" fill="none" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg></div>' +
                '<div class="channel-info">' +
                '<div class="channel-name">' + esc(channelName) + '</div>' +
                '<div class="channel-handle">' + esc(channelUser) + '</div>' +
                (channelPublicLink ? '<div class="channel-link"><a href="' + esc(channelPublicLink) + '" target="_blank">🔗 View Channel</a></div>' : '') +
                '<div class="channel-meta">' +
                '<span class="meta-tag">' + esc(channelCategory) + '</span>' +
                '<span class="meta-tag">' + esc(channelSubs) + ' subs</span>' +
                '<span class="meta-tag">' + esc(channelViews) + ' views</span>' +
                '</div>' +
                (hasSuccessRate ? '<div class="channel-success">⭐ Success Rate: ' + esc(toText(channel.success_rate)) + '%</div>' : '') +
                '</div>' +
                '<div class="channel-price">' +
                '<div class="price-value">' + esc(toText(channelPrice)) + ' TON</div>' +
                '<div class="price-label">per post</div>' +
                '</div>' +
                '</div>';
        });
        container.innerHTML = html;

        // Bind channel selection
        container.querySelectorAll('.channel-card').forEach(function (card) {
            card.addEventListener('click', function () {
                var id = this.dataset ? toText(this.dataset.id, '').trim() : '';
                if (!id) {
                    return;
                }
                var idx = State.selected.indexOf(id);
                if (idx === -1) {
                    State.selected.push(id);
                    this.classList.add('selected');
                } else {
                    State.selected.splice(idx, 1);
                    this.classList.remove('selected');
                }
                console.log('Selected channels:', State.selected);
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

    async function uploadMedia(file) {
        const MAX_UPLOAD_SIZE = 50 * 1024 * 1024;

        if (file.size > MAX_UPLOAD_SIZE) {
            alert("Maximum allowed file size is 50MB");
            return null;
        }

        var formData = new FormData();
        formData.append("file", file);

        var res = await fetch('/api/upload', {method:'POST',body:formData});
        var data = await res.json();
        if (!data.success) {
            alert(data.error);
            return null;
        }
        return data.data;
    }

    // Submit advertising request
    async function submitRequest() {
        var titleEl = document.getElementById('campTitle');
        var textEl = document.getElementById('campText');
        var budgetEl = document.getElementById('campBudget');

        var title = toText(titleEl && titleEl.value).trim();
        var text = toText(textEl && textEl.value).trim();
        var budget = toNumber(budgetEl && budgetEl.value, 0);

        if (safeArray(State.selected).length === 0) {
            toast('Please select at least one channel before submitting.', 'error');
            return;
        }

        if (!title || title.length < 3) {
            toast('Please enter a valid campaign title', 'error');
            return;
        }
        if (!text || text.length < 10) {
            toast('Please enter advertisement text (min 10 characters)', 'error');
            return;
        }
        if (budget < 5) {
            toast('Minimum budget is 5 TON', 'error');
            return;
        }

        setLoading('btnSubmitRequest', true);

        try {
            const telegramId = Number(window.Telegram.WebApp.initDataUnsafe.user.id);

            if (!Number.isFinite(telegramId) || telegramId <= 0) {
                toast("Telegram authentication failed. Reopen Mini App.", "error");
                return;
            }

            var campaignData = {
                title: title,
                text: text,
                budget: budget
            };

            var fileInput = document.getElementById('mediaUpload');
            if (fileInput && fileInput.files.length > 0) {
                var upload = await uploadMedia(fileInput.files[0]);
                if (!upload) {
                    setLoading('btnSubmitRequest', false);
                    return;
                }

                campaignData.media_type = upload.media_type;
                campaignData.media_url = upload.media_url;
            }

            var campaignPayload = {
                title: campaignData.title,
                text: campaignData.text,
                budget: Number(campaignData.budget),
                telegram_id: telegramId
            };
            console.log('Campaign payload:', campaignPayload);

            var res = await apiPost('/api/campaign/create', campaignPayload);

            var createdCampaign = safeObj(res && res.data);

            if (!createdCampaign || !createdCampaign.id) {
                throw new Error("Campaign creation response missing id");
            }
            if (!(res && res.success === true && createdCampaign && createdCampaign.id)) {
                throw new Error(res && res.error ? res.error : 'Failed to create campaign');
            }
            State.campaign = createdCampaign;

            var selected = safeArray(State.selected).map(function (channelId) {
                return toText(channelId, '').trim();
            }).filter(function (channelId) {
                return !!channelId;
            });

            var dealPromises = selected.map(function (channelId) {
                if (!channelId) {
                    throw new Error('Invalid channel_id selected: ' + channelId);
                }

                var channel = safeArray(State.channels).find(function (c) {
                    var item = safeObj(c);
                    return toText(item.id, '').trim() === channelId;
                });

                var amount = toNumber(channel && channel.price, 0);
                if (!(amount > 0)) {
                    throw new Error('Invalid channel amount for channel #' + channelId);
                }

                var selectedCampaign = { id: createdCampaign.id };
                var selectedChannel = { id: channelId };
                return submitDeal(selectedCampaign, selectedChannel, amount);
            });

            await Promise.all(dealPromises);
            showConfirmation();
            toast('Request submitted successfully', 'success');
        } catch (e) {
            console.log('Error submitting request:', e);
            toast('Error submitting request: ' + (e && e.message ? e.message : 'Unknown error'), 'error');
        } finally {
            setLoading('btnSubmitRequest', false);
        }
    }

    // Show confirmation screen
    function showConfirmation() {
        var total = 0;
        State.selected.forEach(function (id) {
            var ch = State.channels.find(function (c) {
                return toText(safeObj(c).id, '').trim() === id;
            });
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
        var mediaUpload = document.getElementById('mediaUpload');
        if (mediaUpload) mediaUpload.value = '';
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
        var ownerWallet = document.getElementById('ownerWallet').value.trim();

        if (!handle) { toast('Please enter channel username', 'error'); return; }
        if (!handle.startsWith('@')) handle = '@' + handle;
        if (!name) { toast('Please enter channel name', 'error'); return; }
        if (!category) { toast('Please select a category', 'error'); return; }
        var allowedCategories = ['crypto', 'nft', 'gaming', 'finance', 'tech', 'Other'];
        if (allowedCategories.indexOf(category) === -1) { toast('Invalid category selected', 'error'); return; }
        if (subs < 2) { toast('Minimum 100 subscribers required', 'error'); return; }
        if (price < 1) { toast('Minimum price is 1 TON', 'error'); return; }
        if (!ownerWallet || (!
            ownerWallet.startsWith('EQ') && !
            ownerWallet.startsWith('UQ'))) {
                alert('Please enter valid TON wallet address');
                return;
         }

        setLoading('btnRegister', true);

        var data = {
            telegram_id: State.user.id,
            username: handle,
            name: name,
            category: category,
            subscribers: subs,
            avg_views: views || Math.floor(subs / 5),
            price: price,
            owner_wallet: ownerWallet
        };

        var authPayload = {
            telegram_id: State.user.id,
            username: toText(State.user && State.user.username, '')
        };

        fetch('/api/auth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(authPayload)
        }).then(function () {
            return fetch('/api/register-channel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
        }).then(function (res) {
            return res.json();
        }).then(function (json) {
            if (json.success) {
                toast('Channel registered successfully', 'success');
                clearChannelForm();
                loadChannels(); // Reload channels list
                return;
            }

            console.error('Channel register failed:', json.error);
            toast(json.error || 'Failed to register channel', 'error');
        }).catch(function (err) {
            console.error('Fetch error:', err);
            toast('Error registering channel', 'error');
        }).finally(function () {
            setLoading('btnRegister', false);
        });

        console.log('register_channel payload', { action: 'register_channel', data: data });
    }

    // Clear channel form
    function clearChannelForm() {
        ['chHandle', 'chName', 'chCategory', 'chSubs', 'chViews', 'chPrice', 'ownerWallet'].forEach(function (id) {
            document.getElementById(id).value = '';
        });
    }

    // Load deals from backend
    async function loadDeals() {
        var telegramId = State.user && State.user.id ? Number(State.user.id) : 0;
        var channelId = State.user && (State.user.channelId || State.user.channel_id || null);

        if (!Number.isFinite(telegramId) || telegramId <= 0) {
            toast('Telegram authentication failed', 'error');
            State.deals = [];
            renderDeals([]);
            return;
        }

        var url = '/api/deals?telegram_id=' + encodeURIComponent(telegramId);
        if (channelId) {
            url += '&user_channel_uid=' + encodeURIComponent(channelId);
        }

        try {
            var res = await fetch(url);
            var response = await res.json();

            if (!response.success) {
                console.error('Failed to fetch deals', response.error);
                State.deals = [];
                renderDeals([]);
                return;
            }

            State.deals = safeArray(response.data);
            renderDeals(response.data);
        } catch (e) {
            console.log('Error loading deals:', e);
            State.deals = [];
            renderDeals([]);
        }
    }

    function renderDeals(deals) {
      const container = document.getElementById("dealList");
      if (!container) {
        console.error("dealList container not found");
        return;
      }
      container.innerHTML = "";

      if (!deals || deals.length === 0) {
        container.innerHTML = "<p>No deals found</p>";
        return;
      }

      deals.forEach(deal => {

        const card = document.createElement("div");
        card.className = "deal-card";

        const statusLabel = deal.label || deal.status || "pending";

        let buttonsHTML = "";

        if (Array.isArray(deal.allowed_actions) && deal.allowed_actions.length > 0) {
          deal.allowed_actions.forEach(action => {

            const label = action
              .split("_")
              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
              .join(" ");

            buttonsHTML += `
          <button 
            class="deal-action-btn"
            data-id="${deal.id}"
            data-action="${action}">
            ${label}
          </button>
        `;
          });
        }

        card.innerHTML = `
      <h3>Deal #${deal.id}</h3>
      <p><strong>Amount:</strong> ${deal.amount || ""} TON</p>
      <p><strong>Status:</strong> ${statusLabel}</p>
      <div class="deal-buttons">
        ${buttonsHTML}
      </div>
    `;

        container.appendChild(card);
      });

      attachDealActionHandlers();
    }

    function attachDealActionHandlers() {
      document.querySelectorAll(".deal-action-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {

          const dealId = e.target.dataset.id;
          const action = e.target.dataset.action;

          try {
            await apiPost("/api/deal/action", {
              deal_id: dealId,
              action: action
            });

            await loadDeals();

          } catch (err) {
            alert("Action failed");
            console.error(err);
          }
        });
      });
    }



    function loadLeaderboard() {
        var container = document.getElementById('leaderboardList');
        if (!container) return;

        container.innerHTML = '<div class="empty"><div class="empty-text">Loading leaderboard...</div></div>';

        apiGet('/api/leaderboard/monthly')
            .then(function (res) {
                var leaders = safeArray(res && res.data && res.data.leaders);
                if (!leaders.length) {
                    container.innerHTML = '<div class="empty"><div class="empty-text">No completed deals this month yet.<br>Be the first to earn! 🚀</div></div>';
                    return;
                }

                container.innerHTML = '';
                leaders.forEach(function (item) {
                    var rank = toNumber(item.rank, 0);
                    var rankClass = '';
                    if (rank === 1) rankClass = 'rank-1';
                    else if (rank === 2) rankClass = 'rank-2';
                    else if (rank === 3) rankClass = 'rank-3';

                    var row = document.createElement('div');
                    row.className = 'leaderboard-row';

                    row.innerHTML =
                        '<div class="rank-badge ' + rankClass + '">' +
                        (rank === 1 ? '👑 ' : '') +
                        '#' + esc(rank) +
                        '</div>' +
                        '<div class="leader-info">' +
                        '<div>@' + esc(toText(item.username, 'unknown')) + '</div>' +
                        '<div class="leader-meta">' +
                        esc(toText(item.total_earned, 0)) + ' TON • ' +
                        esc(toText(item.completed_count, 0)) + ' deals • ⭐ ' +
                        esc(toText(item.success_rate, 0)) + '%' +
                        '</div>' +
                        '</div>';

                    row.style.cursor = 'pointer';
                    row.addEventListener('click', function () {
                        if (item.username) {
                            window.open('https://t.me/' + String(item.username).replace('@', ''), '_blank');
                        }
                    });

                    container.appendChild(row);
                });
            })
            .catch(function (e) {
                console.log('Error loading leaderboard:', e);
                container.innerHTML = '<div class="empty"><div class="empty-text">Unable to load leaderboard right now.</div></div>';
            });
    }

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
    async function apiPost(url, body) {
        const session = await ensureAuthenticated();

        if (!session?.access_token) {
            throw new Error("User not authenticated");
        }

        const res = await fetch(API_BASE + url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${session.access_token}`
            },
            body: JSON.stringify(body)
        });

        const data = await res.json();
        return data;
    }

    // Debug helper (no-op bot bridge)
    function sendToBot(data) {
        console.log('sendToBot disabled; payload:', data);
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


    // Start application
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(init, 60);
    });

    // Expose state for debugging
    window.AdEscrow = State;
})();
