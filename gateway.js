/*  gateway.js  –  pure LLM proxy + best-LLM selector + loud logs  +  quiz-config routes */
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = process.env.GATEWAY_PORT || process.env.PORT || 3002;

const logFile = path.join(__dirname, 'gateway.log');
function log(...args) {
    const line = `[${new Date().toISOString()}]  ${args.join(' ')}`;
    console.log(line);
    fs.appendFileSync(logFile, line + '\n');
}

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '10mb' }));

/*  ----------  BEST-LLM SELECTOR  (unchanged)  ----------  */
async function pickProvider() {
    const wanted = (process.env.LLM_PROVIDER || 'auto').toLowerCase();
    if (wanted !== 'auto') return wanted;

    log('Auto-detecting best LLM ...');

    // 1.  Ollama (fastest local)
    try {
        const r = await fetch('http://127.0.0.1:11434/api/tags', { timeout: 2000 });
        if (r.ok) { log('✓ Ollama detected'); return 'ollama'; }
    } catch { /* ignore */ }

    // 2.  Llama.cpp (local fallback)
    try {
        const r = await fetch('http://127.0.0.1:8080/health', { timeout: 2000 });
        if (r.ok) { log('✓ Llama.cpp detected'); return 'llama'; }
    } catch { /* ignore */ }

    // 3.  OpenAI (cloud)
    if (process.env.OPENAI_KEY) { log('✓ OpenAI key found'); return 'openai'; }

    // 4.  AWS Bedrock (cloud)
    if (process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY) {
        log('✓ AWS Bedrock credentials found'); return 'bedrock';
    }

    log('✗ No LLM available – using fallback');
    return 'fallback';
}

/*  ----------  LLM CALLERS  (unchanged)  ----------  */
async function callLLM(prompt, provider) {
    log(`Using provider: ${provider}`);
    switch (provider) {
        case 'llama':      return await callLlama(prompt);
        case 'openai':     return await callOpenAI(prompt);
        case 'bedrock':    return await callBedrock(prompt);
        case 'fallback':   return fallbackReport(prompt);
        case 'ollama':     return await callOllama(prompt);
        default:           throw new Error('Unknown provider: ' + provider);
    }
}
async function callOllama(prompt) {
    const url = process.env.OLLAMA_URL || 'http://127.0.0.1:11434/api/generate';
    const model = process.env.OLLAMA_MODEL || 'mistral';
    log('Ollama request', { model, promptLen: prompt.length });
    const body = { model, prompt, stream: false, options: { temperature: 0.7, num_predict: 4000 } };
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
    const j = await res.json();
    const clean = j.response?.replace(/\s+/g, ' ').trim();
    log(`Ollama (${model}) clean length: ${clean.length}`);
    return clean;
}
async function callLlama(prompt) {
    const url = process.env.LLAMA_URL || 'http://127.0.0.1:8080/completion';
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, temperature: 0.7, n_predict: 4000, stop: ["</s>", "###", "Human:", "AI:"] }) });
    if (!res.ok) throw new Error(`Llama.cpp HTTP ${res.status}`);
    const j = await res.json();
    return j.content?.replace(/\s+/g, ' ').trim();
}
async function callOpenAI(prompt) {
    const url = 'https://api.openai.com/v1/chat/completions';
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.OPENAI_KEY}` }, body: JSON.stringify({ model: process.env.OPENAI_MODEL || 'gpt-3.5-turbo', messages: [{ role: 'user', content: prompt }], temperature: 0.7, max_tokens: 4000 }) });
    if (!res.ok) throw new Error(`OpenAI HTTP ${res.status}`);
    const j = await res.json();
    return j.choices?.[0]?.message?.content?.replace(/\s+/g, ' ').trim();
}
async function callBedrock(prompt) {
    /*  placeholder – install @aws-sdk/client-bedrock-runtime  */
    throw new Error('Bedrock not implemented – add your code here');
}
function fallbackReport(prompt) {
    return `
        <h2>AI Maturity Analysis</h2>
        <p><strong>Executive Summary:</strong> Your organisation has started the AI journey; focus on the lowest-scoring categories first.</p>
        <h3>Priority Improvements</h3>
        <ul>
            <li>Run AI literacy workshops</li>
            <li>Build basic data pipelines</li>
            <li>Pick 1–2 low-risk pilots</li>
        </ul>
        <h3>12-Month Roadmap</h3>
        <ul>
            <li>Q1: Data audit + pilot selection</li>
            <li>Q2: Pilot deployment + talent upskilling</li>
            <li>Q3: Scale successful pilots + MLOps</li>
            <li>Q4: Governance + ROI measurement</li>
        </ul>
        <p><em>Connect a local Llama.cpp server or set OPENAI_KEY for deeper insights.</em></p>
    `;
}

/*  ----------  EXISTING ENDPOINTS  (unchanged)  ----------  */
app.get('/api/health', (req, res) => {
    log('GET /api/health');
    res.json({ status: 'OK', ts: new Date().toISOString() });
});

app.post('/api/analyse', async (req, res) => {
    log('POST /api/analyse  body length:', JSON.stringify(req.body).length);
    try {
        const { prompt } = req.body;
        if (!prompt || typeof prompt !== 'string') {
            log('Missing or invalid prompt');
            return res.status(400).json({ analysis: '<p>Missing or invalid prompt.</p>' });
        }
        const provider = await pickProvider();
        const raw = await callLLM(prompt, provider);
        log('LLM response length:', raw.length);
        res.json({ analysis: raw });
    } catch (err) {
        log('Gateway error:', err.message);
        res.status(500).json({ analysis: `<p>Gateway error: ${err.message}</p>` });
    }
});

app.post('/api/pdf', async (req, res) => {
    /*  optional – leave stub for now  */
    res.status(501).json({ msg: 'PDF route not enabled locally' });
});

/*  ----------  NEW CONFIG ENDPOINTS  (platform only)  ----------  */
app.get('/api/config', (req, res) => {
    const cfg = JSON.parse(fs.readFileSync('./public/config/quizzes.json', 'utf8'));
    res.json(cfg);                 // full catalog
});

app.get('/api/config/:quizId', (req, res) => {
    const cfg = JSON.parse(fs.readFileSync('./public/config/quizzes.json', 'utf8'));
    const quiz = cfg.find(q => q.id === req.params.quizId);
    if (!quiz) return res.status(404).json({ error: 'Quiz not found' });
    res.json(quiz);                // single quiz incl. promptBuilder
});

/*  ----------  START SERVER  ----------  */
app.listen(PORT, () => console.log(`Gateway running → http://localhost:${PORT}`));
