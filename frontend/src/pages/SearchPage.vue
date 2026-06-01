<template>
  <div class="page">
    <header class="topbar">
      <span class="logo">🏥 医疗智能问答</span>
      <span class="sub">Hybrid Search · RRF · Rerank · RAG</span>
    </header>

    <section class="search-section">
      <div class="search-box">
        <textarea v-model="query" class="search-input" placeholder="输入您的健康问题，例如：高烧不退怎么办？" rows="3" @keydown.ctrl.enter="doSearch" />
        <div class="search-actions">
          <label class="toggle">
            <input type="checkbox" v-model="withAnswer" />
            <span>AI 生成回答</span>
          </label>
          <button class="btn-search" :disabled="loading" @click="doSearch">
            {{ loading ? '检索中…' : '提问' }}
          </button>
        </div>
      </div>
      <p class="hint">Ctrl + Enter 快速提交</p>
    </section>

    <div v-if="error" class="error-bar">{{ error }}</div>

    <main v-if="result" class="result-section">
      <div v-if="result.llm_answer" class="llm-card">
        <div class="llm-label"><span class="dot" />AI 综合回答</div>
        <p class="llm-text">{{ result.llm_answer }}</p>
        <p class="llm-disclaimer">⚠️ 仅供参考，不构成医疗建议，如有不适请及时就医。</p>
      </div>

      <div class="refs-header">
        <span>参考文档（共 {{ result.results.length }} 条）</span>
        <span class="score-tip">相关度由高到低</span>
      </div>

      <div v-for="(doc, idx) in result.results" :key="doc.doc_id" class="ref-card">
        <div class="ref-head" @click="toggle(idx)">
          <span class="ref-num">{{ idx + 1 }}</span>
          <span class="ref-question">{{ doc.question }}</span>
          <span class="ref-score">{{ (doc.score * 100).toFixed(1) }}</span>
          <span class="ref-arrow">{{ expandedIdx === idx ? '▲' : '▼' }}</span>
        </div>
        <div v-show="expandedIdx === idx" class="ref-body">
          <div class="ref-section">
            <span class="ref-tag">问题</span>
            <p>{{ doc.question }}</p>
          </div>
          <div class="ref-section">
            <span class="ref-tag">医生回答</span>
            <p class="ref-answer">{{ doc.answer }}</p>
          </div>
          <div class="ref-meta">匹配片段：{{ doc.chunk_snippet }}</div>
        </div>
      </div>
    </main>

    <div v-else-if="!loading" class="empty-state">
      <p>输入症状或医学问题，获取参考回答</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { search } from '../api/search.js'

const query = ref('')
const withAnswer = ref(true)
const loading = ref(false)
const error = ref('')
const result = ref(null)
const expandedIdx = ref(0)

async function doSearch() {
  if (!query.value.trim()) return
  loading.value = true
  error.value = ''
  result.value = null
  expandedIdx.value = 0
  try {
    const res = await search(query.value.trim(), 8, withAnswer.value)
    result.value = res.data
  } catch (e) {
    error.value = e.response?.data?.detail || '请求失败，请检查后端服务是否启动'
  } finally {
    loading.value = false
  }
}

function toggle(idx) {
  expandedIdx.value = expandedIdx.value === idx ? -1 : idx
}
</script>

<style scoped>
.page { max-width:860px; margin:0 auto; padding:0 16px 60px; }
.topbar { display:flex; align-items:baseline; gap:12px; padding:20px 0 12px; border-bottom:1px solid var(--border); margin-bottom:32px; }
.logo { font-size:1.25rem; font-weight:700; color:var(--primary); }
.sub { font-size:0.8rem; color:var(--muted); }
.search-box { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:16px; }
.search-input { width:100%; border:none; outline:none; resize:none; font-size:1rem; line-height:1.6; color:var(--text); background:transparent; }
.search-actions { display:flex; align-items:center; justify-content:space-between; margin-top:12px; }
.toggle { display:flex; align-items:center; gap:6px; font-size:0.85rem; color:var(--muted); cursor:pointer; }
.btn-search { background:var(--primary); color:#fff; border:none; border-radius:8px; padding:8px 28px; font-size:0.95rem; cursor:pointer; }
.btn-search:hover:not(:disabled) { background:var(--primary-h); }
.btn-search:disabled { opacity:.6; cursor:not-allowed; }
.hint { font-size:0.75rem; color:var(--muted); margin-top:6px; }
.error-bar { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:12px 16px; color:#b91c1c; margin-bottom:16px; }
.llm-card { background:#eff6ff; border:1px solid #bfdbfe; border-radius:var(--radius); padding:20px 24px; margin-bottom:24px; }
.llm-label { display:flex; align-items:center; gap:8px; font-weight:600; color:var(--primary); margin-bottom:12px; }
.dot { width:8px; height:8px; border-radius:50%; background:var(--primary); }
.llm-text { line-height:1.75; white-space:pre-wrap; }
.llm-disclaimer { margin-top:12px; font-size:0.8rem; color:var(--muted); }
.refs-header { display:flex; justify-content:space-between; font-size:0.85rem; color:var(--muted); margin-bottom:12px; }
.ref-card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:10px; overflow:hidden; }
.ref-head { display:flex; align-items:center; gap:10px; padding:14px 16px; cursor:pointer; }
.ref-head:hover { background:#f8fafc; }
.ref-num { width:24px; height:24px; background:var(--primary); color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; flex-shrink:0; }
.ref-question { flex:1; font-size:0.95rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ref-score { font-size:0.8rem; color:var(--primary); font-weight:600; margin-right:4px; }
.ref-arrow { color:var(--muted); font-size:0.7rem; }
.ref-body { padding:0 16px 16px; border-top:1px solid var(--border); }
.ref-section { margin-top:14px; }
.ref-tag { display:inline-block; background:#e0f2fe; color:#0369a1; border-radius:4px; padding:1px 8px; font-size:0.75rem; margin-bottom:6px; }
.ref-answer { line-height:1.7; white-space:pre-wrap; }
.ref-meta { margin-top:12px; padding:10px 12px; background:var(--bg); border-radius:6px; font-size:0.8rem; color:var(--muted); }
.empty-state { text-align:center; color:var(--muted); padding:80px 0; }
</style>
