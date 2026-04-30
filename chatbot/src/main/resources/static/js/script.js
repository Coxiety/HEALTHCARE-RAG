"use strict";

document.addEventListener('DOMContentLoaded', () => {

    // ==========================================
    // KHAI BÁO DOM ELEMENTS
    // ==========================================
    const sidebar           = document.getElementById('sidebar');
    const menuToggle        = document.getElementById('menu-toggle');
    const searchHistoryBtn  = document.getElementById('search-history-btn');
    const chatInput         = document.getElementById('chat-input');
    const sendBtn           = document.getElementById('send-btn');
    const welcomeArea       = document.getElementById('welcome-area');
    const messagesWrapper   = document.getElementById('messages-wrapper');
    const chatContainer     = document.getElementById('chat-container');
    const avatarBtn         = document.getElementById('avatar-btn');
    const authModal         = document.getElementById('auth-modal');
    const closeModal        = document.getElementById('close-modal');
    const tabLogin          = document.getElementById('tab-login');
    const tabRegister       = document.getElementById('tab-register');
    const loginForm         = document.getElementById('login-form');
    const registerForm      = document.getElementById('register-form');
    const profilePopup      = document.getElementById('profile-popup');
    const accountList       = document.getElementById('account-list');
    const dashboardPanel    = document.getElementById('dashboard-panel');
    const dashboardToggle   = document.getElementById('dashboard-toggle-btn');
    const closeDashboard    = document.getElementById('close-dashboard');
    const newChatBtn        = document.getElementById('new-chat-btn');

    // Cấu hình marked.js
    if (typeof marked !== 'undefined') {
        marked.setOptions({ breaks: true, gfm: true });
    }

    // ==========================================
    // MODULE 1: TƯƠNG TÁC UI CƠ BẢN
    // ==========================================

    menuToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));

    if (searchHistoryBtn) {
        searchHistoryBtn.addEventListener('click', () => {
            alert('Tính năng Tìm kiếm đoạn chat đang được phát triển!');
        });
    }

    // New chat — session mới, reset màn hình welcome
    newChatBtn.addEventListener('click', () => {
        localStorage.setItem('sessionId', crypto.randomUUID());
        messagesWrapper.innerHTML = '';
        if (welcomeArea) welcomeArea.style.display = 'block';
        loadSessions();
    });

    // Tự động thay đổi chiều cao Textarea
    chatInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
        sendBtn.classList.toggle('active', this.value.trim() !== '');
    });

    // ==========================================
    // MODULE 2: QUẢN LÝ TÀI KHOẢN & AUTH
    // ==========================================

    function updateAvatarUI() {
        const username = localStorage.getItem('username');
        const src = username
            ? `https://ui-avatars.com/api/?name=${encodeURIComponent(username)}&background=a8c7fa&color=131314`
            : `https://ui-avatars.com/api/?name=User&background=random`;
        avatarBtn.innerHTML = `<img src="${src}" alt="Avatar">`;
    }

    function updateGreeting() {
        const fullName = localStorage.getItem('fullName');
        const el = document.getElementById('greeting-headline');
        if (!el) return;
        if (fullName) {
            const hour = new Date().getHours();
            const greeting = hour < 12 ? 'Chào buổi sáng' : hour < 18 ? 'Chào buổi chiều' : 'Chào buổi tối';
            el.textContent = `${greeting}, ${fullName}!`;
        } else {
            el.textContent = 'Xin chào!';
        }
    }

    updateAvatarUI();
    updateGreeting();

    avatarBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (localStorage.getItem('token')) {
            renderProfilePopup();
            profilePopup.classList.toggle('active');
            authModal.classList.remove('active');
        } else {
            authModal.classList.add('active');
            profilePopup.classList.remove('active');
        }
    });

    function renderProfilePopup() {
        const currentUsername = localStorage.getItem('username');
        const currentFullName = localStorage.getItem('fullName') || currentUsername;
        document.getElementById('popup-avatar').src =
            `https://ui-avatars.com/api/?name=${encodeURIComponent(currentUsername)}&background=a8c7fa&color=131314`;
        document.getElementById('popup-fullname').innerText = `Chào ${currentFullName},`;

        const sessions = JSON.parse(localStorage.getItem('sessions')) || [];
        accountList.innerHTML = '';
        sessions.forEach(session => {
            if (session.username === currentUsername) return;
            const div = document.createElement('div');
            div.className = 'account-item';
            div.innerHTML = `
                <img src="https://ui-avatars.com/api/?name=${encodeURIComponent(session.username)}&background=random" alt="Avatar">
                <div class="account-info">
                    <span class="acc-name">${session.fullName || session.username}</span>
                    <span class="acc-user">${session.username}</span>
                </div>`;
            div.onclick = () => {
                localStorage.setItem('token', session.token);
                localStorage.setItem('userId', session.userId);
                localStorage.setItem('username', session.username);
                localStorage.setItem('fullName', session.fullName);
                localStorage.removeItem('sessionId');
                updateAvatarUI();
                updateGreeting();
                profilePopup.classList.remove('active');
                messagesWrapper.innerHTML = '';
                if (welcomeArea) welcomeArea.style.display = 'block';
                loadSessions();
            };
            accountList.appendChild(div);
        });
    }

    document.getElementById('add-account-btn').addEventListener('click', () => {
        profilePopup.classList.remove('active');
        authModal.classList.add('active');
    });

    document.getElementById('logout-btn').addEventListener('click', () => {
        localStorage.clear();
        updateAvatarUI();
        updateGreeting();
        profilePopup.classList.remove('active');
        messagesWrapper.innerHTML = '';
        if (welcomeArea) welcomeArea.style.display = 'block';
        document.getElementById('chat-history-list').innerHTML = '';
    });

    document.addEventListener('click', (e) => {
        if (!profilePopup.contains(e.target) && !avatarBtn.contains(e.target)) {
            profilePopup.classList.remove('active');
        }
    });

    closeModal.addEventListener('click', () => authModal.classList.remove('active'));

    tabLogin.addEventListener('click', () => {
        tabLogin.classList.add('active'); tabRegister.classList.remove('active');
        loginForm.classList.add('active'); registerForm.classList.remove('active');
    });

    tabRegister.addEventListener('click', () => {
        tabRegister.classList.add('active'); tabLogin.classList.remove('active');
        registerForm.classList.add('active'); loginForm.classList.remove('active');
    });

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fullName = document.getElementById('reg-fullname').value;
        const username = document.getElementById('reg-username').value;
        const password = document.getElementById('reg-password').value;
        const msgEl = document.getElementById('reg-msg');
        try {
            const res = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fullName, username, password })
            });
            const text = await res.text();
            if (res.ok) {
                msgEl.className = 'auth-msg success-msg';
                msgEl.innerText = text;
                setTimeout(() => tabLogin.click(), 1500);
            } else {
                msgEl.className = 'auth-msg error-msg';
                msgEl.innerText = text;
            }
        } catch {
            msgEl.className = 'auth-msg error-msg';
            msgEl.innerText = 'Lỗi kết nối Server!';
        }
    });

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;
        const msgEl = document.getElementById('login-msg');
        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            if (res.ok) {
                const data = await res.json();
                let sessions = JSON.parse(localStorage.getItem('sessions')) || [];
                sessions = sessions.filter(s => s.username !== data.username);
                sessions.push(data);
                localStorage.setItem('sessions', JSON.stringify(sessions));
                localStorage.setItem('token', data.token);
                localStorage.setItem('userId', data.userId);
                localStorage.setItem('username', data.username);
                localStorage.setItem('fullName', data.fullName || data.username);
                msgEl.className = 'auth-msg success-msg';
                msgEl.innerText = 'Đăng nhập thành công!';
                setTimeout(async () => {
                    authModal.classList.remove('active');
                    updateAvatarUI();
                    updateGreeting();
                    loginForm.reset();
                    msgEl.innerText = '';
                    await loadSessions();
                    await loadChatHistory();
                }, 800);
            } else {
                msgEl.className = 'auth-msg error-msg';
                msgEl.innerText = 'Tài khoản hoặc mật khẩu không chính xác!';
            }
        } catch {
            msgEl.innerHTML = '<span class="error-msg">Lỗi kết nối Server!</span>';
        }
    });

    // ==========================================
    // MODULE 3: INTENT HELPERS
    // ==========================================

    function intentToClass(intent) {
        const map = {
            'NUTRITION_LOOKUP': 'nutrition',
            'HEALTH_ADVICE': 'health',
            'BOTH': 'both',
            'GREETING': 'greeting'
        };
        return map[intent] || 'nutrition';
    }

    function intentToLabel(intent) {
        const map = {
            'NUTRITION_LOOKUP': 'Tra cứu dinh dưỡng',
            'HEALTH_ADVICE': 'Tư vấn sức khỏe',
            'BOTH': 'Dinh dưỡng & Sức khỏe',
            'GREETING': 'Chào hỏi'
        };
        return map[intent] || intent;
    }

    function intentToIcon(intent) {
        const map = {
            'NUTRITION_LOOKUP': 'nutrition',
            'HEALTH_ADVICE': 'medical_services',
            'BOTH': 'spa',
            'GREETING': 'waving_hand'
        };
        return map[intent] || 'spa';
    }

    // ==========================================
    // MODULE 4: CHAT BUBBLE BUILDERS
    // ==========================================

    function createMessageBubble(sender, content, isError = false) {
        const msgRow = document.createElement('div');
        msgRow.className = `msg-row ${sender}`;
        if (sender === 'user') {
            msgRow.innerHTML = `<div class="msg-bubble">${escapeHtml(content)}</div>`;
        } else {
            const color = isError ? 'var(--color-error)' : 'var(--color-accent)';
            const icon = isError ? 'error' : 'spa';
            msgRow.innerHTML = `
                <div class="msg-bubble">
                    <div class="msg-header">
                        <span class="material-symbols-outlined" style="color:${color}">${icon}</span>
                    </div>
                    <div class="msg-text" style="${isError ? 'color:var(--color-error)' : ''}">${content}</div>
                </div>`;
        }
        return msgRow;
    }

    function createRichAiBubble(data) {
        const msgRow = document.createElement('div');
        msgRow.className = 'msg-row ai';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';

        // 1. Header: icon + intent badge
        if (data.intent) {
            const cls = intentToClass(data.intent);
            const label = intentToLabel(data.intent);
            const icon = intentToIcon(data.intent);
            const header = document.createElement('div');
            header.className = 'msg-header';
            header.innerHTML = `
                <span class="material-symbols-outlined" style="color:var(--color-accent);font-size:20px">${icon}</span>
                <span class="intent-badge intent-${cls}">${label}</span>`;
            bubble.appendChild(header);
        }

        // 2. Nội dung trả lời (markdown)
        const msgText = document.createElement('div');
        msgText.className = 'msg-text';
        if (typeof marked !== 'undefined') {
            msgText.innerHTML = marked.parse(data.answer || '');
        } else {
            msgText.innerHTML = (data.answer || '').replace(/\n/g, '<br>');
        }
        bubble.appendChild(msgText);

        // 3. Nutrition card (nếu có energy)
        if (data.energy && data.energy.amountPer100g != null) {
            const card = document.createElement('div');
            card.className = 'nutrition-card';
            card.innerHTML = `
                <span class="material-symbols-outlined" style="color:var(--color-success);font-size:28px">local_fire_department</span>
                <div>
                    <div class="nutrition-value">${data.energy.amountPer100g}</div>
                    <div class="nutrition-label">${data.energy.unitName || 'kcal'} / 100g</div>
                </div>`;
            bubble.appendChild(card);
        }

        // 4. Sources
        if (data.sources && data.sources.length > 0) {
            const srcSection = document.createElement('div');
            srcSection.className = 'sources-section';
            srcSection.innerHTML = `
                <div class="sources-title">
                    <span class="material-symbols-outlined">source</span>Nguồn tham khảo
                </div>`;
            data.sources.forEach(src => {
                const item = document.createElement('div');
                item.className = 'source-item';
                item.innerHTML = `<span class="material-symbols-outlined">article</span>${escapeHtml(src)}`;
                srcSection.appendChild(item);
            });
            bubble.appendChild(srcSection);
        }

        msgRow.appendChild(bubble);
        return msgRow;
    }

    function createTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'msg-row ai loading-msg';
        row.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>`;
        return row;
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function scrollToBottom() {
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    }

    // ==========================================
    // MODULE 5: SESSION MANAGEMENT
    // ==========================================

    function getOrCreateSession() {
        let sid = localStorage.getItem('sessionId');
        if (!sid) {
            sid = crypto.randomUUID();
            localStorage.setItem('sessionId', sid);
        }
        return sid;
    }

    function renderMessages(messages) {
        messages.forEach(msg => {
            if (msg.role === 'user') {
                messagesWrapper.appendChild(createMessageBubble('user', msg.content));
            } else {
                let parsedEntities = {};
                let parsedSources  = [];
                try { if (msg.entitiesJson) parsedEntities = JSON.parse(msg.entitiesJson); } catch {}
                try { if (msg.sourcesJson)  parsedSources  = JSON.parse(msg.sourcesJson);  } catch {}
                messagesWrapper.appendChild(createRichAiBubble({
                    answer:   msg.content,
                    intent:   msg.intent,
                    entities: parsedEntities,
                    sources:  parsedSources,
                    energy:   msg.energyAmount != null
                        ? { amountPer100g: msg.energyAmount, unitName: msg.energyUnit }
                        : null
                }));
            }
        });
    }

    async function loadSessions() {
        const token  = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');
        const list   = document.getElementById('chat-history-list');
        if (!token || !userId || !list) return;

        try {
            const res = await fetch(`/api/chat/sessions/${userId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.status === 401 || res.status === 403) {
                localStorage.removeItem('token');
                localStorage.removeItem('userId');
                updateAvatarUI();
                updateGreeting();
                return;
            }
            if (!res.ok) return;
            const sessions = await res.json();
            list.innerHTML = '';
            sessions.forEach(s => {
                const btn = document.createElement('button');
                btn.className = 'nav-item chat-history-item';
                btn.innerHTML = `
                    <span class="material-symbols-outlined">chat_bubble</span>
                    <span class="nav-text">${escapeHtml(s.title)}</span>`;
                btn.onclick = () => loadSession(s.sessionId);
                list.appendChild(btn);
            });
        } catch (err) {
            console.warn('Không tải được sessions:', err);
        }
    }

    async function loadSession(sessionId) {
        const token = localStorage.getItem('token');
        if (!token) return;
        try {
            const res = await fetch(`/api/chat/session/${sessionId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!res.ok) return;
            const messages = await res.json();
            localStorage.setItem('sessionId', sessionId);
            messagesWrapper.innerHTML = '';
            if (welcomeArea) welcomeArea.style.display = 'none';
            renderMessages(messages);
            scrollToBottom();
        } catch (err) {
            console.warn('Không tải được session:', err);
        }
    }

    // ==========================================
    // MODULE 6: CHAT HISTORY (load lần đầu đăng nhập)
    // ==========================================

    async function loadChatHistory() {
        const token  = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');
        if (!token || !userId) return;

        try {
            const res = await fetch(`/api/chat/history/${userId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.status === 401 || res.status === 403) {
                localStorage.removeItem('token');
                localStorage.removeItem('userId');
                updateAvatarUI();
                updateGreeting();
                return;
            }
            if (!res.ok) return;
            const messages = await res.json();
            if (!messages || messages.length === 0) return;

            if (welcomeArea) welcomeArea.style.display = 'none';
            renderMessages(messages);
            scrollToBottom();
        } catch (err) {
            console.warn('Không tải được lịch sử chat:', err);
        }
    }

    // Tải sessions và history nếu đã đăng nhập sẵn
    if (localStorage.getItem('token')) {
        loadSessions();
        loadChatHistory();
    }

    // ==========================================
    // MODULE 7: DASHBOARD PANEL
    // ==========================================

    dashboardToggle.addEventListener('click', async () => {
        dashboardPanel.classList.toggle('hidden');
        if (!dashboardPanel.classList.contains('hidden')) {
            await loadDashboard();
        }
    });

    closeDashboard.addEventListener('click', () => {
        dashboardPanel.classList.add('hidden');
    });

    async function loadDashboard() {
        const token  = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');
        const list   = document.getElementById('dashboard-list');

        if (!token || !userId) {
            list.innerHTML = '<p class="dash-empty">Vui lòng đăng nhập để xem nhật ký.</p>';
            return;
        }

        list.innerHTML = '<p class="dash-empty">Đang tải...</p>';

        try {
            const res = await fetch(`/api/chat/dashboard/${userId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!res.ok) throw new Error('Lỗi server');
            const records = await res.json();
            list.innerHTML = '';

            if (records.length === 0) {
                list.innerHTML = '<p class="dash-empty">Chưa có dữ liệu dinh dưỡng nào.<br>Hãy hỏi về một món ăn!</p>';
                return;
            }

            records.forEach(r => {
                const d = r.createdAt ? new Date(r.createdAt) : null;
                const dateStr = d ? d.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' }) : '';
                const item = document.createElement('div');
                item.className = 'dash-item';
                item.innerHTML = `
                    <div>
                        <div class="dash-food-name">${escapeHtml(r.foodName || '')}</div>
                        <div class="dash-date">${dateStr}</div>
                    </div>
                    <div class="dash-calories">${escapeHtml(r.calories || '')}</div>`;
                list.appendChild(item);
            });
        } catch (err) {
            list.innerHTML = '<p class="dash-empty">Không tải được dữ liệu.</p>';
        }
    }

    // ==========================================
    // MODULE 8: SEND MESSAGE
    // ==========================================

    async function sendMessage() {
        const text   = chatInput.value.trim();
        const token  = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');

        if (!text) return;

        if (!token) {
            alert('Bạn cần Đăng nhập để AI có thể lưu lại chế độ dinh dưỡng nhé!');
            authModal.classList.add('active');
            return;
        }

        if (welcomeArea) welcomeArea.style.display = 'none';

        messagesWrapper.appendChild(createMessageBubble('user', text));
        scrollToBottom();

        chatInput.value = '';
        chatInput.style.height = 'auto';
        chatInput.disabled = true;
        sendBtn.disabled = true;
        sendBtn.classList.remove('active');

        const loadingMsg = createTypingIndicator();
        messagesWrapper.appendChild(loadingMsg);
        scrollToBottom();

        try {
            const response = await fetch(`/api/chat/send?userId=${userId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ message: text, sessionId: getOrCreateSession() })
            });

            if (response.status === 401 || response.status === 403) {
                localStorage.removeItem('token');
                localStorage.removeItem('userId');
                updateAvatarUI();
                throw new Error('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại!');
            }

            if (!response.ok) throw new Error(`Lỗi máy chủ: ${response.status}`);

            const data = await response.json();
            loadingMsg.remove();

            if (data && data.answer) {
                messagesWrapper.appendChild(createRichAiBubble(data));
                loadSessions(); // cập nhật sidebar nếu đây là tin nhắn đầu của session mới
            } else {
                messagesWrapper.appendChild(createMessageBubble('ai', 'Xin lỗi, tôi chưa hiểu ý bạn.'));
            }

        } catch (error) {
            console.error('Lỗi Chat API:', error);
            loadingMsg.remove();
            messagesWrapper.appendChild(createMessageBubble('ai', error.message || 'Lỗi kết nối đến máy chủ AI!', true));
            if (error.message && error.message.includes('đăng nhập lại')) {
                authModal.classList.add('active');
            }
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            scrollToBottom();
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
