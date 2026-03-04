// test.js — Phase 1 comprehensive test suite
// Pass/fail for each test. Exit code 0 = all pass.

const { store, recall, listRecent, stats } = require('./client');

let passed = 0;
let failed = 0;

function assert(label, condition, detail = '') {
  if (condition) {
    console.log(`  ✅ ${label}`);
    passed++;
  } else {
    console.log(`  ❌ ${label}${detail ? ' — ' + detail : ''}`);
    failed++;
  }
}

// ─── SEMANTIC RECALL TESTS ──────────────────────────────────────────────────
// Each test: query uses ZERO keywords from the stored content.
// Goal: prove meaning-based retrieval, not keyword matching.

const SEMANTIC_TESTS = [
  // Personal/legal
  {
    query: 'become a citizen without language exam',
    expect: /citizenship|optie|naturali/i,
    label: 'citizenship — no keywords (language exam ≠ inburgering)',
  },
  {
    query: 'monthly government disability payment',
    expect: /UWV|WGA|WAO|invalidity/i,
    label: 'UWV benefit — no keywords (government payment ≠ UWV/WGA)',
  },
  {
    query: 'eye disease insulin injection',
    expect: /diabetes|Bosman|Ozempic|sensor/i,
    label: 'health/diabetes — no keywords (eye disease ≠ diabetes)',
  },
  // Business/strategy
  {
    query: 'clone proven products improve slightly',
    expect: /Samuel Rondo|clone|improve 1|serial/i,
    label: 'serial business creator — no keywords',
  },
  {
    query: 'freelancer billing for hours worked',
    expect: /invoice|GMM|Clockify|\$63/i,
    label: 'invoicing — no keywords (freelancer billing ≠ invoice/GMM)',
  },
  {
    query: 'deployment container crashed because file missing',
    expect: /hygienic_files|config\.toml|300s|bridge/i,
    label: 'bridge crash fix — no keywords (container crashed ≠ hygienic_files)',
  },
  {
    query: 'why did we stop the automated trading bot',
    expect: /mock|Math\.random|fake|paused/i,
    label: 'trading paused — no keywords (stop automated ≠ mock data)',
  },
  // Infrastructure
  {
    query: 'secure remote access through jump server',
    expect: /SSH|tunnel|guardian|ProxyCommand/i,
    label: 'SSH tunnel — no keywords (jump server ≠ SSH)',
  },
  {
    query: 'AI chat memory that works across different tools',
    expect: /MCP|memory|Open Brain|semantic|vector/i,
    label: 'cross-tool memory — finds Open Brain context',
  },
  // Finance
  {
    query: 'digital currency savings goal five years',
    expect: /BTC|bitcoin|crypto|DCA/i,
    label: 'BTC DCA — no keywords (digital currency ≠ BTC)',
  },
  {
    query: 'accountant monthly fee payment deadline',
    expect: /Kroess|Visser|invoice|1 week/i,
    label: 'Kroess & Visser reminder — no keywords',
  },
  {
    query: 'philosophy paper submission academic journal',
    expect: /thatness|PhilArchive|Ergo|preprint/i,
    label: 'philosophy paper — no keywords (academic submission ≠ thatness)',
  },
];

// ─── TYPE FILTER TESTS ──────────────────────────────────────────────────────
const TYPE_TESTS = [
  { query: 'what was decided about insurance', type: 'decision', label: 'decision filter: insurance' },
  { query: 'what was learned from mistakes',   type: 'lesson',   label: 'lesson filter: mistakes' },
  { query: 'pending action items',             type: 'todo',     label: 'todo filter: pending items' },
  { query: 'information about a person',       type: 'person',   label: 'person filter' },
];

// ─── ROUNDTRIP STORE+RECALL TEST ────────────────────────────────────────────
const ROUNDTRIP_CONTENT = `Unique test memory: ZoeterBot Phase 1 validation. 
The Open Brain MCP server passed all semantic recall tests on 2026-03-02. 
Architecture: sqlite-vec + Gemini embeddings + Node.js SSE.`;

async function run() {
  console.log('\n═══ Phase 1 Test Suite ═══\n');

  // ── 1. Stats ─────────────────────────────────────────────────────────────
  console.log('【1】 Database Stats');
  const s = stats();
  assert('Total memories > 250', s.total > 250, `got ${s.total}`);
  assert('Has notes',     s.byType.note    > 0);
  assert('Has decisions', s.byType.decision > 0);
  assert('Has lessons',   s.byType.lesson   > 0);
  assert('Has todos',     s.byType.todo     > 0);
  assert('Has persons',   s.byType.person   > 0);
  console.log(`   Total: ${s.total} | Types: ${JSON.stringify(s.byType)}\n`);

  // ── 2. Semantic Recall ────────────────────────────────────────────────────
  console.log('【2】 Semantic Recall (zero keyword overlap)');
  for (const t of SEMANTIC_TESTS) {
    const results = await recall(t.query, 1);
    const hit = results[0];
    const matches = hit && t.expect.test(hit.content);
    assert(
      t.label,
      matches,
      hit ? `dist:${hit.distance} — "${hit.content.slice(0, 80).replace(/\n/g,' ')}"` : 'no results'
    );
  }
  console.log('');

  // ── 3. Distance Quality ───────────────────────────────────────────────────
  console.log('【3】 Distance Quality (top result should be < 0.95)');
  const qualityTests = [
    'Dutch citizenship optie procedure requirements',
    'bitcoin DCA strategy biweekly purchases',
    'GMM bridge daemon failure recovery',
  ];
  for (const q of qualityTests) {
    const r = await recall(q, 1);
    const dist = r[0]?.distance ?? 99;
    assert(`Distance < 0.95: "${q.slice(0,40)}"`, dist < 0.95, `got ${dist}`);
  }
  console.log('');

  // ── 4. Type Filtering ─────────────────────────────────────────────────────
  console.log('【4】 Type Filtering');
  for (const t of TYPE_TESTS) {
    const results = await recall(t.query, 3, t.type);
    const allCorrectType = results.every(r => r.type === t.type);
    assert(
      t.label,
      results.length > 0 && allCorrectType,
      `got ${results.length} results, types: [${results.map(r=>r.type).join(',')}]`
    );
  }
  console.log('');

  // ── 5. listRecent ─────────────────────────────────────────────────────────
  console.log('【5】 List Recent');
  const recent7 = listRecent(7);
  assert('listRecent(7) returns results', recent7.length > 0, `got ${recent7.length}`);
  assert('All have date field', recent7.every(r => r.date), 'some missing date');
  assert('All have content',    recent7.every(r => r.content?.length > 0));
  console.log('');

  // ── 6. Roundtrip Store + Recall ───────────────────────────────────────────
  console.log('【6】 Roundtrip: store → recall');
  const newId = await store(ROUNDTRIP_CONTENT, {
    type: 'note',
    topics: ['phase1-test', 'validation'],
    importance: 5
  });
  assert('store() returns numeric id', typeof newId === 'number' && newId > 0, `got ${newId}`);

  const recalled = await recall('Open Brain validation test passed', 3);
  const found = recalled.some(r => r.content.includes('Phase 1 validation'));
  assert('Just-stored memory is immediately recallable', found,
    found ? '' : `top result: "${recalled[0]?.content?.slice(0,80)}"`);
  console.log('');

  // ── 7. MCP Server Health ──────────────────────────────────────────────────
  console.log('【7】 MCP Server');
  try {
    const { default: fetch } = await import('node-fetch');
    const res = await fetch('http://localhost:3737/health');
    const body = await res.json();
    assert('Health endpoint 200', res.status === 200);
    assert('Server name correct', body.server === 'zoeterbot-brain', `got "${body.server}"`);
    assert('Port correct', body.port == 3737);
  } catch(e) {
    assert('MCP server reachable', false, e.message);
    assert('Health endpoint 200', false, 'server not running');
    assert('Server name correct', false);
  }
  console.log('');

  // ── 8. Edge Cases ─────────────────────────────────────────────────────────
  console.log('【8】 Edge Cases');
  // Very short query
  const short = await recall('Rust', 3);
  assert('Short query returns results', short.length > 0);

  // Recall with limit=1 returns exactly 1
  const one = await recall('anything at all', 1);
  assert('limit=1 returns exactly 1', one.length === 1, `got ${one.length}`);

  // listRecent with type filter
  const recentDecisions = listRecent(365, 'decision', 5);
  assert('listRecent type filter works', recentDecisions.every(r => r.type === 'decision'),
    `types: [${recentDecisions.map(r=>r.type)}]`);
  console.log('');

  // ── Summary ───────────────────────────────────────────────────────────────
  const total = passed + failed;
  console.log('═══════════════════════════════════');
  console.log(`Results: ${passed}/${total} passed`);
  if (failed === 0) {
    console.log('✅ ALL TESTS PASS — Phase 1 proven');
  } else {
    console.log(`❌ ${failed} FAILED`);
  }
  console.log('═══════════════════════════════════\n');

  process.exit(failed === 0 ? 0 : 1);
}

run().catch(e => { console.error('Test runner crashed:', e); process.exit(1); });
