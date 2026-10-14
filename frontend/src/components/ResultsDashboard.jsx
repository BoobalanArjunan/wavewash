import { useState, useMemo } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { Download, CheckCircle2, Wrench, XCircle, Copy, ChevronDown, ChevronRight, FolderOpen, AlertTriangle } from 'lucide-react'
import { downloadUrl, openFolder } from '../api'

const STATUS_META = {
  passed: { label: 'Passed clean', color: 'var(--color-pass)', icon: CheckCircle2 },
  fixed: { label: 'Repaired & passed', color: 'var(--color-fixed)', icon: Wrench },
  failed: { label: 'Rejected', color: 'var(--color-fail)', icon: XCircle },
  duplicate: { label: 'Duplicate removed', color: 'var(--color-dup)', icon: Copy },
}

function StatCard({ statusKey, count, total }) {
  const meta = STATUS_META[statusKey]
  if (!meta) return null
  const Icon = meta.icon
  const pct = total > 0 ? Math.round((count / total) * 100) : 0
  return (
    <div className="flex-1 bg-[var(--color-panel)] border border-[var(--color-border)] rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <Icon size={15} style={{ color: meta.color }} />
        <span className="text-xs text-[var(--color-text-muted)]">{meta.label}</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold font-mono-data">{count}</span>
        <span className="text-xs text-[var(--color-text-faint)] font-mono-data">{pct}%</span>
      </div>
    </div>
  )
}

function FileRow({ file }) {
  const [open, setOpen] = useState(false)
  const meta = STATUS_META[file.status] || { color: 'var(--color-text-faint)', icon: XCircle }
  const Icon = meta.icon
  const hasDetail = file.issues_found || file.fixes_applied || file.unfixable_reasons || file.warnings || file.duplicate_of

  return (
    <div className="border-b border-[var(--color-border)] last:border-0">
      <button
        onClick={() => hasDetail && setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-[var(--color-panel-raised)] transition-colors text-left"
      >
        {hasDetail ? (
          open ? <ChevronDown size={13} className="text-[var(--color-text-faint)] shrink-0" /> : <ChevronRight size={13} className="text-[var(--color-text-faint)] shrink-0" />
        ) : <div className="w-[13px]" />}
        <Icon size={14} style={{ color: meta.color }} className="shrink-0" />
        <span className="text-sm font-mono-data truncate flex-1">{file.filename}</span>
        {file.warnings && <AlertTriangle size={13} className="text-[var(--color-fixed)] shrink-0" title="Passed with a note" />}
        {file.snr_db && <span className="text-xs font-mono-data text-[var(--color-text-faint)] hidden sm:block">SNR {parseFloat(file.snr_db).toFixed(1)}dB</span>}
      </button>
      {open && hasDetail && (
        <div className="px-4 pb-3 pl-10 space-y-1.5">
          {file.duplicate_of && (
            <div className="text-xs">
              <span className="text-[var(--color-text-faint)]">Same recording as: </span>
              <span className="font-mono-data text-[var(--color-dup)]">{file.duplicate_of.split(/[\\/]/).pop()}</span>
              <span className="text-[var(--color-text-faint)]"> (copied to _duplicates/; your original is untouched)</span>
            </div>
          )}
          {file.warnings && (
            <div className="text-xs">
              <span className="text-[var(--color-text-faint)]">Note: </span>
              <span className="font-mono-data text-[var(--color-fixed)]">{file.warnings}</span>
            </div>
          )}
          {file.issues_found && !file.duplicate_of && (
            <div className="text-xs">
              <span className="text-[var(--color-text-faint)]">Found: </span>
              <span className="font-mono-data text-[var(--color-fail)]">{file.issues_found}</span>
            </div>
          )}
          {file.fixes_applied && (
            <div className="text-xs">
              <span className="text-[var(--color-text-faint)]">Fixed: </span>
              <span className="font-mono-data text-[var(--color-fixed)]">{file.fixes_applied}</span>
            </div>
          )}
          {file.unfixable_reasons && (
            <div className="text-xs">
              <span className="text-[var(--color-text-faint)]">Rejected because: </span>
              <span className="font-mono-data text-[var(--color-fail)]">{file.unfixable_reasons}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ResultsDashboard({ results, jobId, onReset }) {
  const [filter, setFilter] = useState('all')

  const total = Object.values(results.status_counts || {}).reduce((a, b) => a + b, 0)
  const pieData = useMemo(() =>
    Object.entries(results.status_counts || {}).map(([key, value]) => ({
      name: STATUS_META[key]?.label || key, value, color: STATUS_META[key]?.color || '#666',
    })), [results])

  const filteredFiles = filter === 'all'
    ? results.files
    : results.files.filter(f => f.status === filter)

  return (
    <div className="h-full overflow-y-auto p-6">
      {results.error && (
        <div className="mb-6 p-4 rounded-lg border border-[var(--color-fail)]/40 bg-[var(--color-fail)]/10">
          <div className="text-sm font-semibold text-[var(--color-fail)] mb-1">The run didn’t finish</div>
          <pre className="text-[11px] font-mono-data text-[var(--color-text-muted)] whitespace-pre-wrap max-h-48 overflow-y-auto">{results.log_tail || 'No log output was captured.'}</pre>
        </div>
      )}
      <div className="flex items-start justify-between mb-6">
        <div className="min-w-0">
          <h2 className="text-lg font-bold">{results.error ? 'Run failed' : 'Run complete'}</h2>
          <p className="text-sm text-[var(--color-text-muted)] mt-0.5">{total} files processed. Your original files were not changed.</p>
          {results.base_dir && (
            <button onClick={() => openFolder(results.base_dir).catch(() => {})}
              className="mt-1.5 flex items-center gap-1.5 text-xs font-mono-data text-[var(--color-accent)] hover:underline max-w-full">
              <FolderOpen size={12} className="shrink-0" /><span className="truncate">{results.base_dir}</span>
            </button>
          )}
        </div>
        <div className="flex gap-2">
          <a
            href={downloadUrl(jobId)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-[#14161A] font-semibold text-sm hover:brightness-110 transition-all"
          >
            <Download size={14} /> Download all results
          </a>
          <button
            onClick={onReset}
            className="px-4 py-2 rounded-lg border border-[var(--color-border)] text-sm hover:bg-[var(--color-panel-raised)] transition-colors"
          >
            New run
          </button>
        </div>
      </div>

      <div className="flex gap-3 mb-6">
        <div className="flex gap-3 flex-1 flex-wrap">
          {Object.entries(results.status_counts || {}).map(([key, count]) => (
            <StatCard key={key} statusKey={key} count={count} total={total} />
          ))}
        </div>
        {pieData.length > 0 && (
          <div className="w-32 h-32 shrink-0 hidden md:block">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" innerRadius={30} outerRadius={55} paddingAngle={2}>
                  {pieData.map((entry, i) => <Cell key={i} fill={entry.color} stroke="none" />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#1C1F24', border: '1px solid #2A2E35', borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setFilter('all')} className={`text-xs px-3 py-1.5 rounded-full transition-colors ${filter === 'all' ? 'bg-[var(--color-accent)] text-[#14161A] font-semibold' : 'bg-[var(--color-panel)] text-[var(--color-text-muted)] hover:bg-[var(--color-panel-raised)]'}`}>
          All ({total})
        </button>
        {Object.entries(results.status_counts || {}).map(([key, count]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`text-xs px-3 py-1.5 rounded-full transition-colors ${filter === key ? 'bg-[var(--color-accent)] text-[#14161A] font-semibold' : 'bg-[var(--color-panel)] text-[var(--color-text-muted)] hover:bg-[var(--color-panel-raised)]'}`}
          >
            {STATUS_META[key]?.label || key} ({count})
          </button>
        ))}
      </div>

      <div className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded-lg overflow-hidden">
        {filteredFiles.map((f, i) => <FileRow key={i} file={f} />)}
      </div>
    </div>
  )
}
