import { useEffect, useState, useCallback, useRef } from 'react';
import { ref, listAll, getDownloadURL, getMetadata } from 'firebase/storage';
import { storage } from './firebase';

// ── Constants & Utilities ──────────────────────────────────────────────────

const ANIMAL_KEYWORDS = [
  'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
  'giraffe','person','human','deer','fox','rabbit','squirrel','raccoon',
  'coyote','hawk','owl','turkey','heron','wolf','moose','beaver',
];

const ANIMAL_ALIASES = { human: 'person' };

function extractAnimal(filename = '') {
  const parts = filename.toLowerCase().split(/[_\-\s\d\.]+/);
  for (const part of parts) {
    if (ANIMAL_KEYWORDS.includes(part)) return ANIMAL_ALIASES[part] || part;
  }
  return null;
}

function extractConfidence(filename = '') {
  const match = filename.match(/conf(0\.\d+)/i);
  return match ? parseFloat(match[1]) : null;
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  const h = Math.floor(diff / 3600000);
  const d = Math.floor(diff / 86400000);
  if (m < 1)  return 'just now';
  if (m < 60) return `${m}m ago`;
  if (h < 24) return `${h}h ago`;
  return `${d}d ago`;
}

function getTimeOfDay(iso) {
  const h = new Date(iso).getHours();
  if (h >= 5  && h < 12) return 'morning';
  if (h >= 12 && h < 17) return 'afternoon';
  if (h >= 17 && h < 21) return 'evening';
  return 'night';
}

const TIME_OF_DAY_OPTIONS = [
  { value: 'all',       label: '🌐 All Day' },
  { value: 'morning',   label: '🌅 Morning (5–12)' },
  { value: 'afternoon', label: '☀️ Afternoon (12–17)' },
  { value: 'evening',   label: '🌆 Evening (17–21)' },
  { value: 'night',     label: '🌙 Night (21–5)' },
];

const BADGE_COLORS = {
  bird:'#4ade80', cat:'#f472b6', dog:'#fb923c', horse:'#a78bfa',
  sheep:'#67e8f9', cow:'#fbbf24', elephant:'#6ee7b7', bear:'#f87171',
  zebra:'#e2e8f0', giraffe:'#fde68a', person:'#93c5fd', deer:'#fcd34d',
  fox:'#fb923c', rabbit:'#d8b4fe', squirrel:'#a3e635', raccoon:'#94a3b8',
  hawk:'#fdba74', owl:'#c4b5fd', turkey:'#86efac', heron:'#67e8f9',
  wolf:'#f1f5f9', moose:'#d97706', beaver:'#b45309', default:'#86efac',
};

function getBadgeColor(animal) {
  return BADGE_COLORS[animal] || BADGE_COLORS.default;
}

// ── Chart Components (pure SVG, no dependencies) ──────────────────────────

function DonutChart({ data, total }) {
  if (!data.length) return null;
  const size = 140, cx = 70, cy = 70, r = 52, stroke = 18;
  const circumference = 2 * Math.PI * r;
  let offset = 0;
  const slices = data.map(([animal, count]) => {
    const pct   = count / total;
    const dash  = pct * circumference;
    const slice = { animal, count, pct, dash, offset };
    offset += dash;
    return slice;
  });

  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: 160 }}>
      {slices.map((s) => (
        <circle key={s.animal} cx={cx} cy={cy} r={r}
          fill="none"
          stroke={getBadgeColor(s.animal)}
          strokeWidth={stroke}
          strokeDasharray={`${s.dash} ${circumference - s.dash}`}
          strokeDashoffset={-s.offset + circumference * 0.25}
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
      ))}
      <text x={cx} y={cy - 6} textAnchor="middle"
        style={{ fill: '#dceedd', fontSize: 20, fontWeight: 700, fontFamily: 'DM Sans' }}>
        {total}
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle"
        style={{ fill: '#7a9e7e', fontSize: 9, fontFamily: 'DM Sans', letterSpacing: 1 }}>
        TOTAL
      </text>
    </svg>
  );
}

function HourBarChart({ images }) {
  const counts = Array(24).fill(0);
  images.forEach(img => counts[new Date(img.created).getHours()]++);
  const max = Math.max(...counts, 1);
  const chartH = 80, barW = 8, gap = 3;
  const totalW = 24 * (barW + gap);

  return (
    <svg viewBox={`0 0 ${totalW} ${chartH + 20}`} style={{ width: '100%' }}>
      {counts.map((c, h) => {
        const barH = (c / max) * chartH;
        const x    = h * (barW + gap);
        const y    = chartH - barH;
        const tod  = getTimeOfDay(new Date(new Date().setHours(h, 0, 0, 0)));
        const color = tod === 'morning' ? '#fde68a'
                    : tod === 'afternoon' ? '#4ade80'
                    : tod === 'evening'   ? '#fb923c'
                    : '#818cf8';
        return (
          <g key={h}>
            <rect x={x} y={y} width={barW} height={barH || 2}
              fill={c > 0 ? color : '#2a3d2b'} rx={2}
              style={{ transition: 'height 0.4s ease, y 0.4s ease' }} />
            {h % 6 === 0 && (
              <text x={x + barW / 2} y={chartH + 14} textAnchor="middle"
                style={{ fill: '#7a9e7e', fontSize: 8, fontFamily: 'DM Sans' }}>
                {h === 0 ? '12a' : h === 12 ? '12p' : h > 12 ? `${h-12}p` : `${h}a`}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function DetectionsLineChart({ images }) {
  if (!images.length) return null;

  const days = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    days.push({ label: d.toLocaleDateString('en-US', { weekday: 'short' }), date: d.toDateString(), count: 0 });
  }
  images.forEach(img => {
    const ds  = new Date(img.created).toDateString();
    const day = days.find(d => d.date === ds);
    if (day) day.count++;
  });

  const max   = Math.max(...days.map(d => d.count), 1);
  const W = 260, H = 80, padL = 20, padB = 18;
  const pts   = days.map((d, i) => ({
    x: padL + (i / 6) * (W - padL - 8),
    y: H - padB - (d.count / max) * (H - padB - 8),
    ...d,
  }));
  const polyline = pts.map(p => `${p.x},${p.y}`).join(' ');
  const area = `${pts[0].x},${H - padB} ${pts.map(p => `${p.x},${p.y}`).join(' ')} ${pts[pts.length-1].x},${H - padB}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }}>
      <defs>
        <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#4ade80" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#4ade80" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.5, 1].map(pct => {
        const y = H - padB - pct * (H - padB - 8);
        return <line key={pct} x1={padL} y1={y} x2={W - 8} y2={y}
          stroke="#2a3d2b" strokeWidth="1" strokeDasharray="3 3" />;
      })}
      <polygon points={area} fill="url(#lineGrad)" />
      <polyline points={polyline} fill="none"
        stroke="#4ade80" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r={3} fill="#4ade80" />
          <text x={p.x} y={H - 4} textAnchor="middle"
            style={{ fill: '#7a9e7e', fontSize: 8, fontFamily: 'DM Sans' }}>
            {p.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

function ConfidenceHistogram({ images }) {
  // Only images that have a confidence value
  const withConf = images.filter(img => img.confidence !== null && img.confidence !== undefined);
  if (!withConf.length) return (
    <p style={{ color: '#7a9e7e', fontSize: '0.8rem', marginTop: 8 }}>
      No confidence data yet — filenames need <code>conf0.XX</code>.
    </p>
  );

  // 10 buckets: 0–10%, 10–20%, … 90–100%
  const buckets = Array(10).fill(0);
  withConf.forEach(img => {
    const i = Math.min(Math.floor(img.confidence * 10), 9);
    buckets[i]++;
  });
  const max    = Math.max(...buckets, 1);
  const chartH = 80, barW = 18, gap = 4;
  const totalW = 10 * (barW + gap);

  // Color each bar: red → yellow → green based on confidence bucket
  const barColor = (i) => {
    if (i >= 8) return '#4ade80';   // 80–100% green
    if (i >= 5) return '#fbbf24';   // 50–80% yellow
    return '#f87171';               // 0–50%  red
  };

  return (
    <svg viewBox={`0 0 ${totalW} ${chartH + 20}`} style={{ width: '100%' }}>
      {buckets.map((c, i) => {
        const barH = (c / max) * chartH;
        const x    = i * (barW + gap);
        const y    = chartH - barH;
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={barH || 2}
              fill={c > 0 ? barColor(i) : '#2a3d2b'} rx={3}
              style={{ transition: 'height 0.4s ease, y 0.4s ease' }} />
            {c > 0 && (
              <text x={x + barW / 2} y={y - 3} textAnchor="middle"
                style={{ fill: '#7a9e7e', fontSize: 7, fontFamily: 'DM Sans' }}>
                {c}
              </text>
            )}
            <text x={x + barW / 2} y={chartH + 14} textAnchor="middle"
              style={{ fill: '#7a9e7e', fontSize: 7, fontFamily: 'DM Sans' }}>
              {i * 10}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [allImages, setAllImages]         = useState([]);
  const [images, setImages]               = useState([]);
  const [loading, setLoading]             = useState(true);
  const [selectedImage, setSelectedImage] = useState(null);
  const [currentFolder, setCurrentFolder] = useState('detected');
  const [searchQuery, setSearchQuery]     = useState('');
  const [dateFrom, setDateFrom]           = useState('');
  const [dateTo, setDateTo]               = useState('');
  const [timeOfDay, setTimeOfDay]         = useState('all');
  const [autoRefresh, setAutoRefresh]     = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [refreshing, setRefreshing]       = useState(false);
  const [newCount, setNewCount]           = useState(0);
  const [notes, setNotes]                 = useState(() => {
    try { return JSON.parse(localStorage.getItem('ww_notes') || '{}'); }
    catch { return {}; }
  });
  const [editingNote, setEditingNote]     = useState(null);
  const [noteDraft, setNoteDraft]         = useState('');
  const [ticker, setTicker]               = useState(null);
  const intervalRef  = useRef(null);
  const prevCountRef = useRef(0);

  // ── Persist notes ──────────────────────────────────────────────────────
  useEffect(() => {
    localStorage.setItem('ww_notes', JSON.stringify(notes));
  }, [notes]);

  // ── Fetch ──────────────────────────────────────────────────────────────
  const fetchImages = useCallback(async (folder, isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);

    const fetchAll = async (dirRef) => {
      let files = [];
      const res = await listAll(dirRef);
      files = [...files, ...res.items];
      for (const sub of res.prefixes) files = [...files, ...await fetchAll(sub)];
      return files;
    };

    try {
      const itemRefs = await fetchAll(ref(storage, `${folder}/`));
      const data = await Promise.all(
        itemRefs.map(async (item) => {
          const [url, meta] = await Promise.all([getDownloadURL(item), getMetadata(item)]);
          return { url, name: item.name, created: meta.timeCreated, animal: extractAnimal(item.name), confidence: extractConfidence(item.name) };
        })
      );
      const sorted = data.sort((a, b) => new Date(b.created) - new Date(a.created));

      if (isRefresh && sorted.length > prevCountRef.current) {
        const added = sorted.length - prevCountRef.current;
        setNewCount(added);
        setTimeout(() => setNewCount(0), 8000);
        if (sorted[0]) setTicker(sorted[0]);
      }
      prevCountRef.current = sorted.length;

      setAllImages(sorted);
      setLastRefreshed(new Date());
      if (!isRefresh && sorted[0]) setTicker(sorted[0]);
    } catch (err) {
      console.error('Error loading images:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { fetchImages(currentFolder); }, [currentFolder, fetchImages]);

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => fetchImages(currentFolder, true), 30000);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [autoRefresh, currentFolder, fetchImages]);

  // ── Filter ─────────────────────────────────────────────────────────────
  useEffect(() => {
    let f = [...allImages];
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      f = f.filter(img => img.name.toLowerCase().includes(q) || (img.animal || '').includes(q));
    }
    if (dateFrom) f = f.filter(img => new Date(img.created) >= new Date(dateFrom));
    if (dateTo) {
      const end = new Date(dateTo); end.setHours(23, 59, 59, 999);
      f = f.filter(img => new Date(img.created) <= end);
    }
    if (timeOfDay !== 'all') f = f.filter(img => getTimeOfDay(img.created) === timeOfDay);
    setImages(f);
  }, [allImages, searchQuery, dateFrom, dateTo, timeOfDay]);

  // ── Stats ──────────────────────────────────────────────────────────────
  const animalCounts = allImages.reduce((acc, img) => {
    const k = img.animal || 'unknown';
    acc[k] = (acc[k] || 0) + 1;
    return acc;
  }, {});
  const topAnimals = Object.entries(animalCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const todayCount = allImages.filter(
    img => new Date(img.created).toDateString() === new Date().toDateString()
  ).length;
  const hasFilters = searchQuery || dateFrom || dateTo || timeOfDay !== 'all';

  // ── Keyboard ───────────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') { setSelectedImage(null); setEditingNote(null); } };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);

  // ── Download ───────────────────────────────────────────────────────────
  const downloadImage = (img, e) => {
    e.stopPropagation();
    window.open(img.url, '_blank');
  };

  // ── Notes ──────────────────────────────────────────────────────────────
  const saveNote = (name) => {
    setNotes(prev => ({ ...prev, [name]: noteDraft }));
    setEditingNote(null);
  };
  const deleteNote = (name) => {
    setNotes(prev => { const n = { ...prev }; delete n[name]; return n; });
  };

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <>
      <style>{CSS}</style>
      <div className="app">

        {/* Notification toast */}
        {newCount > 0 && (
          <div className="toast">
            🔔 {newCount} new image{newCount > 1 ? 's' : ''} arrived!
          </div>
        )}

        {/* Header */}
        <header className="header">
          <div className="header-inner">
            <div className="logo-block">
              <span className="logo-icon">🦎</span>
              <div>
                <h1 className="site-title">Real-Time Wildlife Detection Feed</h1>
                <p className="site-sub">Capstone 2026</p>
              </div>
            </div>
            <div className="header-controls">
              <button
                className={`refresh-btn ${autoRefresh ? 'active' : ''}`}
                onClick={() => setAutoRefresh(v => !v)}>
                <span className={`spin-icon ${autoRefresh ? 'spinning' : ''}`}>⟳</span>
                {autoRefresh ? 'Live' : 'Auto-Refresh'}
              </button>
              <button className="refresh-btn"
                onClick={() => fetchImages(currentFolder, true)} disabled={refreshing}>
                {refreshing ? 'Refreshing…' : 'Refresh Now'}
              </button>
            </div>
          </div>
          {lastRefreshed && (
            <p className="last-refreshed">Last updated: {formatDate(lastRefreshed.toISOString())}</p>
          )}
          {/* Live ticker */}
          {ticker && (
            <div className="ticker">
              <span className="ticker-dot" />
              <span className="ticker-label">Last seen:</span>
              {ticker.animal && (
                <span className="ticker-badge"
                  style={{ background: getBadgeColor(ticker.animal), color: '#1a2e1a' }}>
                  {ticker.animal}
                </span>
              )}
              <span className="ticker-name">{ticker.name}</span>
              <span className="ticker-time">{timeAgo(ticker.created)}</span>
            </div>
          )}
        </header>

        {/* Charts row */}
        {!loading && allImages.length > 0 && (
          <div className="charts-row">
            <div className="chart-card">
              <h3 className="chart-title">Detections — Last 7 Days</h3>
              <DetectionsLineChart images={allImages} />
            </div>
            <div className="chart-card">
              <h3 className="chart-title">Activity by Hour</h3>
              <HourBarChart images={allImages} />
              <div className="hour-legend">
                {[['#fde68a','Morning'],['#4ade80','Afternoon'],['#fb923c','Evening'],['#818cf8','Night']].map(([c,l]) => (
                  <span key={l} className="hour-legend-item">
                    <span style={{ background: c, width: 8, height: 8, borderRadius: 2, display: 'inline-block', marginRight: 4 }} />
                    {l}
                  </span>
                ))}
              </div>
            </div>
            <div className="chart-card donut-card">
              <h3 className="chart-title">Species Breakdown</h3>
              <div className="donut-wrap">
                <DonutChart data={topAnimals} total={allImages.length} />
                <div className="donut-legend">
                  {topAnimals.map(([animal, count]) => (
                    <div key={animal} className="donut-legend-item">
                      <span className="badge-dot" style={{ background: getBadgeColor(animal) }} />
                      <span style={{ textTransform: 'capitalize' }}>{animal}</span>
                      <span className="animal-count">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="chart-card">
              <h3 className="chart-title">Confidence Distribution</h3>
              <ConfidenceHistogram images={allImages} />
              <div className="hour-legend" style={{ marginTop: 10 }}>
                {[['#f87171','0–50%'],['#fbbf24','50–80%'],['#4ade80','80–100%']].map(([c,l]) => (
                  <span key={l} className="hour-legend-item">
                    <span style={{ background: c, width: 8, height: 8, borderRadius: 2, display: 'inline-block', marginRight: 4 }} />
                    {l}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="layout">
          {/* Sidebar */}
          <aside className="sidebar">

            <section className="sidebar-card">
              <h2 className="sidebar-heading">Overview</h2>
              <div className="stat-row">
                <span className="stat-label">Total Images</span>
                <span className="stat-value">{allImages.length}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Today</span>
                <span className="stat-value highlight">{todayCount}</span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Showing</span>
                <span className="stat-value">{images.length}</span>
              </div>
            </section>

            <section className="sidebar-card">
              <h2 className="sidebar-heading">Folder</h2>
              {['detected', 'empty'].map(f => (
                <button key={f}
                  className={`folder-btn ${currentFolder === f ? 'selected' : ''}`}
                  onClick={() => setCurrentFolder(f)}>
                  {f === 'detected' ? '🎯' : '📭'} {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </section>

            <section className="sidebar-card">
              <h2 className="sidebar-heading">Filters</h2>
              <input className="filter-input" type="text"
                placeholder="Search by animal or filename…"
                value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />

              <label className="filter-label">Time of Day</label>
              <div className="tod-grid">
                {TIME_OF_DAY_OPTIONS.map(opt => (
                  <button key={opt.value}
                    className={`tod-btn ${timeOfDay === opt.value ? 'selected' : ''}`}
                    onClick={() => setTimeOfDay(opt.value)}>
                    {opt.label}
                  </button>
                ))}
              </div>

              <label className="filter-label">From</label>
              <input className="filter-input" type="date"
                value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
              <label className="filter-label">To</label>
              <input className="filter-input" type="date"
                value={dateTo} onChange={e => setDateTo(e.target.value)} />

              {hasFilters && (
                <button className="clear-btn" onClick={() => {
                  setSearchQuery(''); setDateFrom(''); setDateTo(''); setTimeOfDay('all');
                }}>Clear Filters</button>
              )}
            </section>

          </aside>

          {/* Main grid */}
          <main className="main">
            {loading ? (
              <div className="loading-state">
                <div className="loader" />
                <p>Loading feed…</p>
              </div>
            ) : images.length === 0 ? (
              <div className="empty-state">
                <p>🔍 No images match your current filters.</p>
              </div>
            ) : (
              <div className="grid">
                {images.map((img, i) => (
                  <div key={i} className="card"
                    style={{ animationDelay: `${Math.min(i * 40, 400)}ms` }}
                    onClick={() => setSelectedImage(img)}>
                    <div className="card-img-wrap">
                      <img src={img.url} alt="Wildlife Detection" loading="lazy" />
                      {img.animal && (
                        <span className="animal-badge"
                          style={{ background: getBadgeColor(img.animal), color: '#1a2e1a' }}>
                          {img.animal}
                        </span>
                      )}
                      <span className="tod-badge">{getTimeOfDay(img.created)}</span>
                      {img.confidence && (
                        <span className="conf-badge">
                          {(img.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                      <button className="dl-btn" title="Download"
                        onClick={(e) => downloadImage(img, e)}>↓</button>
                    </div>
                    <div className="card-info">
                      <p className="card-date">{formatDate(img.created)}</p>
                      <p className="card-filename">{img.name}</p>
                      {notes[img.name] && (
                        <p className="card-note">📝 {notes[img.name]}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </main>
        </div>

        {/* Modal */}
        {selectedImage && (
          <div className="modal" onClick={() => setSelectedImage(null)}>
            <button className="modal-close" onClick={() => setSelectedImage(null)}>✕</button>
            <div className="modal-inner" onClick={e => e.stopPropagation()}>
              <img className="modal-img" src={selectedImage.url} alt="Enlarged Wildlife" />
              <div className="modal-info">
                <div className="modal-info-top">
                  <div>
                    {selectedImage.animal && (
                      <span className="animal-badge large"
                        style={{ background: getBadgeColor(selectedImage.animal), color: '#1a2e1a' }}>
                        {selectedImage.animal}
                      </span>
                    )}
                    <p className="modal-date">{formatDate(selectedImage.created)}</p>
                    <p className="modal-filename">{selectedImage.name}</p>
                    {selectedImage.confidence && (
                      <p className="modal-conf">
                        Confidence: <strong>{(selectedImage.confidence * 100).toFixed(0)}%</strong>
                      </p>
                    )}
                  </div>
                  <div className="modal-actions">
                    <button className="modal-action-btn"
                      onClick={(e) => downloadImage(selectedImage, e)}>
                      ↓ Download
                    </button>
                    <button className="modal-action-btn note"
                      onClick={() => {
                        setEditingNote(selectedImage.name);
                        setNoteDraft(notes[selectedImage.name] || '');
                      }}>
                      📝 {notes[selectedImage.name] ? 'Edit Note' : 'Add Note'}
                    </button>
                    {notes[selectedImage.name] && (
                      <button className="modal-action-btn danger"
                        onClick={() => deleteNote(selectedImage.name)}>
                        🗑 Remove Note
                      </button>
                    )}
                  </div>
                </div>

                {notes[selectedImage.name] && editingNote !== selectedImage.name && (
                  <div className="note-display">
                    <span className="note-label">Note:</span> {notes[selectedImage.name]}
                  </div>
                )}

                {editingNote === selectedImage.name && (
                  <div className="note-editor">
                    <textarea className="note-textarea" autoFocus rows={3}
                      placeholder="Add a note about this detection…"
                      value={noteDraft} onChange={e => setNoteDraft(e.target.value)} />
                    <div className="note-editor-btns">
                      <button className="note-save-btn" onClick={() => saveNote(selectedImage.name)}>
                        Save Note
                      </button>
                      <button className="note-cancel-btn" onClick={() => setEditingNote(null)}>
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </div>
    </>
  );
}

// ── CSS ────────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:           #0f1a10;
    --surface:      #162018;
    --surface2:     #1d2b1e;
    --border:       #2a3d2b;
    --text:         #dceedd;
    --text-muted:   #7a9e7e;
    --accent:       #4ade80;
    --accent-dim:   #22543d;
    --danger:       #f87171;
    --radius:       12px;
    --font-display: 'Playfair Display', Georgia, serif;
    --font-body:    'DM Sans', sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--font-body); min-height: 100vh; }
  .app { max-width: 1440px; margin: 0 auto; padding: 0 24px 60px; }

  /* Toast */
  .toast {
    position: fixed; top: 20px; right: 24px; z-index: 2000;
    background: var(--accent-dim); border: 1px solid var(--accent); color: var(--accent);
    padding: 12px 20px; border-radius: 10px; font-weight: 600; font-size: 0.9rem;
    animation: slideIn 0.3s ease, fadeOut 0.5s ease 7.5s forwards;
    box-shadow: 0 4px 20px rgba(74,222,128,0.2);
  }
  @keyframes slideIn { from { opacity:0; transform:translateY(-12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes fadeOut { to { opacity:0; } }

  /* Header */
  .header { padding: 28px 0 16px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
  .header-inner { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px; }
  .logo-block { display:flex; align-items:center; gap:16px; }
  .logo-icon { font-size:2.8rem; line-height:1; }
  .site-title { font-family:var(--font-display); font-size:2.2rem; color:var(--accent); letter-spacing:-0.5px; line-height:1.1; }
  .site-sub { color:var(--text-muted); font-size:0.88rem; font-weight:500; margin-top:2px; text-transform:uppercase; letter-spacing:0.06em; }
  .header-controls { display:flex; gap:10px; flex-wrap:wrap; }
  .refresh-btn {
    background:var(--surface2); border:1px solid var(--border); color:var(--text);
    padding:8px 16px; border-radius:8px; cursor:pointer; font-family:var(--font-body);
    font-size:0.88rem; font-weight:500; display:flex; align-items:center; gap:6px; transition:all 0.2s;
  }
  .refresh-btn:hover { border-color:var(--accent); color:var(--accent); }
  .refresh-btn.active { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }
  .refresh-btn:disabled { opacity:0.5; cursor:not-allowed; }
  .spin-icon { font-size:1.1rem; display:inline-block; }
  .spin-icon.spinning { animation:spin 1.2s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .last-refreshed { color:var(--text-muted); font-size:0.78rem; margin-top:8px; }

  /* Ticker */
  .ticker {
    display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    margin-top:14px; padding:10px 16px;
    background:var(--surface); border:1px solid var(--border); border-radius:8px; font-size:0.85rem;
  }
  .ticker-dot { width:8px; height:8px; border-radius:50%; background:var(--accent); flex-shrink:0; animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
  .ticker-label { color:var(--text-muted); font-weight:500; }
  .ticker-badge { padding:2px 8px; border-radius:99px; font-size:0.75rem; font-weight:600; text-transform:capitalize; }
  .ticker-name { color:var(--text); font-weight:500; flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .ticker-time { color:var(--text-muted); font-size:0.78rem; flex-shrink:0; }

  /* Charts */
  .charts-row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:20px; margin-bottom:28px; }
  @media(max-width:1100px){ .charts-row { grid-template-columns:1fr 1fr; } }
  @media(max-width:600px){ .charts-row { grid-template-columns:1fr; } }
  .chart-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px; }
  .chart-title { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); margin-bottom:14px; }
  .hour-legend { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .hour-legend-item { display:flex; align-items:center; font-size:0.72rem; color:var(--text-muted); }
  .donut-card .donut-wrap { display:flex; align-items:center; gap:16px; }
  .donut-legend { flex:1; display:flex; flex-direction:column; gap:6px; }
  .donut-legend-item { display:flex; align-items:center; gap:6px; font-size:0.78rem; }
  .badge-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .animal-count { margin-left:auto; color:var(--text-muted); font-size:0.75rem; }

  /* Layout */
  .layout { display:grid; grid-template-columns:260px 1fr; gap:28px; align-items:start; }
  @media(max-width:860px){ .layout { grid-template-columns:1fr; } }

  /* Sidebar */
  .sidebar { display:flex; flex-direction:column; gap:16px; position:sticky; top:24px; }
  .sidebar-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px; }
  .sidebar-heading { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); margin-bottom:14px; }
  .stat-row { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid var(--border); }
  .stat-row:last-child { border-bottom:none; }
  .stat-label { color:var(--text-muted); font-size:0.88rem; }
  .stat-value { font-weight:600; font-size:1.05rem; }
  .stat-value.highlight { color:var(--accent); }
  .folder-btn {
    display:block; width:100%; text-align:left; background:var(--surface2);
    border:1px solid var(--border); color:var(--text); padding:10px 14px; border-radius:8px;
    cursor:pointer; font-family:var(--font-body); font-size:0.9rem; font-weight:500; margin-bottom:8px; transition:all 0.2s;
  }
  .folder-btn:last-child { margin-bottom:0; }
  .folder-btn:hover { border-color:var(--accent); }
  .folder-btn.selected { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }
  .filter-label { font-size:0.78rem; color:var(--text-muted); display:block; margin:10px 0 4px; }
  .filter-input {
    width:100%; background:var(--surface2); border:1px solid var(--border); color:var(--text);
    padding:9px 12px; border-radius:8px; font-family:var(--font-body); font-size:0.88rem; outline:none; transition:border-color 0.2s;
  }
  .filter-input:focus { border-color:var(--accent); }
  .filter-input::placeholder { color:var(--text-muted); }
  .filter-input[type="date"]::-webkit-calendar-picker-indicator { filter:invert(0.7); cursor:pointer; }
  .tod-grid { display:flex; flex-direction:column; gap:6px; margin-bottom:4px; }
  .tod-btn {
    width:100%; text-align:left; background:var(--surface2); border:1px solid var(--border);
    color:var(--text-muted); padding:7px 10px; border-radius:7px; cursor:pointer;
    font-family:var(--font-body); font-size:0.82rem; font-weight:500; transition:all 0.2s;
  }
  .tod-btn:hover { border-color:var(--accent); color:var(--text); }
  .tod-btn.selected { background:var(--accent-dim); border-color:var(--accent); color:var(--accent); }
  .clear-btn {
    margin-top:12px; width:100%; background:transparent; border:1px solid var(--danger);
    color:var(--danger); padding:8px; border-radius:8px; cursor:pointer;
    font-family:var(--font-body); font-size:0.84rem; font-weight:500; transition:all 0.2s;
  }
  .clear-btn:hover { background:rgba(248,113,113,0.1); }

  /* Grid */
  .main { min-width:0; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px,1fr)); gap:20px; }
  .card {
    background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
    overflow:hidden; cursor:pointer;
    transition:transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    animation:fadeUp 0.45s ease both;
  }
  @keyframes fadeUp { from{opacity:0;transform:translateY(16px);} to{opacity:1;transform:translateY(0);} }
  .card:hover { transform:translateY(-4px); box-shadow:0 12px 32px rgba(0,0,0,0.4); border-color:var(--accent); }
  .card-img-wrap { position:relative; }
  .card-img-wrap img { width:100%; height:210px; object-fit:cover; display:block; background:var(--surface2); }
  .animal-badge {
    position:absolute; top:10px; left:10px; padding:4px 10px; border-radius:99px;
    font-size:0.75rem; font-weight:600; text-transform:capitalize; letter-spacing:0.04em;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);
  }
  .animal-badge.large { position:static; font-size:0.9rem; padding:6px 14px; display:inline-block; margin-bottom:8px; }
  .tod-badge {
    position:absolute; bottom:8px; left:10px; background:rgba(0,0,0,0.55); color:var(--text-muted);
    padding:2px 8px; border-radius:99px; font-size:0.7rem; font-weight:500;
    text-transform:capitalize; backdrop-filter:blur(4px);
  }
  .dl-btn {
    position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.6);
    border:1px solid rgba(255,255,255,0.15); color:#fff; width:30px; height:30px;
    border-radius:50%; font-size:1rem; cursor:pointer; display:flex; align-items:center; justify-content:center;
    opacity:0; transition:opacity 0.2s; backdrop-filter:blur(4px);
  }
  .card:hover .dl-btn { opacity:1; }
  .dl-btn:hover { background:var(--accent); color:#1a2e1a; border-color:var(--accent); }
  .card-info { padding:14px; }
  .card-date { font-size:0.82rem; font-weight:600; color:var(--text); margin-bottom:4px; }
  .card-filename { font-size:0.75rem; color:var(--text-muted); word-break:break-all; line-height:1.4; }
  .card-note { font-size:0.75rem; color:var(--accent); margin-top:6px; font-style:italic; line-height:1.4; }

  .conf-badge {
    position:absolute; bottom:8px; right:10px;
    background:rgba(0,0,0,0.55); color:#fbbf24;
    padding:2px 8px; border-radius:99px;
    font-size:0.7rem; font-weight:600;
    backdrop-filter:blur(4px);
  }
  .modal-conf { font-size:0.85rem; color:var(--text-muted); margin-top:6px; }
  .modal-conf strong { color:#fbbf24; }

  /* States */
  .loading-state, .empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:300px; color:var(--text-muted); gap:16px; }
  .loader { width:36px; height:36px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; }

  /* Modal */
  .modal { position:fixed; inset:0; background:rgba(0,0,0,0.92); z-index:1000; display:flex; align-items:center; justify-content:center; padding:24px; animation:fadeIn 0.2s ease; }
  @keyframes fadeIn { from{opacity:0;} to{opacity:1;} }
  .modal-close { position:absolute; top:20px; right:24px; background:var(--surface); border:1px solid var(--border); color:var(--text); width:40px; height:40px; border-radius:50%; font-size:1.1rem; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:all 0.2s; }
  .modal-close:hover { background:var(--danger); border-color:var(--danger); }
  .modal-inner { max-width:900px; width:100%; background:var(--surface); border:1px solid var(--border); border-radius:16px; overflow:hidden; max-height:90vh; overflow-y:auto; }
  .modal-img { width:100%; max-height:55vh; object-fit:contain; background:#0a120b; display:block; }
  .modal-info { padding:18px 22px; border-top:1px solid var(--border); }
  .modal-info-top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap; margin-bottom:12px; }
  .modal-date { font-size:0.88rem; font-weight:600; color:var(--text); margin-bottom:4px; margin-top:4px; }
  .modal-filename { font-size:0.8rem; color:var(--text-muted); word-break:break-all; }
  .modal-actions { display:flex; flex-direction:column; gap:8px; flex-shrink:0; }
  .modal-action-btn { background:var(--surface2); border:1px solid var(--border); color:var(--text); padding:7px 14px; border-radius:8px; cursor:pointer; font-family:var(--font-body); font-size:0.82rem; font-weight:500; white-space:nowrap; transition:all 0.2s; }
  .modal-action-btn:hover { border-color:var(--accent); color:var(--accent); }
  .modal-action-btn.note:hover { border-color:#fbbf24; color:#fbbf24; }
  .modal-action-btn.danger:hover { border-color:var(--danger); color:var(--danger); }
  .note-display { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:10px 14px; font-size:0.85rem; line-height:1.5; color:var(--text); }
  .note-label { color:var(--text-muted); font-weight:600; margin-right:6px; }
  .note-editor { margin-top:12px; }
  .note-textarea { width:100%; background:var(--surface2); border:1px solid var(--border); color:var(--text); padding:10px 12px; border-radius:8px; font-family:var(--font-body); font-size:0.88rem; outline:none; resize:vertical; transition:border-color 0.2s; }
  .note-textarea:focus { border-color:var(--accent); }
  .note-editor-btns { display:flex; gap:8px; margin-top:8px; }
  .note-save-btn { background:var(--accent-dim); border:1px solid var(--accent); color:var(--accent); padding:7px 16px; border-radius:8px; cursor:pointer; font-family:var(--font-body); font-size:0.84rem; font-weight:600; transition:all 0.2s; }
  .note-save-btn:hover { background:var(--accent); color:#1a2e1a; }
  .note-cancel-btn { background:transparent; border:1px solid var(--border); color:var(--text-muted); padding:7px 16px; border-radius:8px; cursor:pointer; font-family:var(--font-body); font-size:0.84rem; transition:all 0.2s; }
  .note-cancel-btn:hover { border-color:var(--text-muted); color:var(--text); }
`;