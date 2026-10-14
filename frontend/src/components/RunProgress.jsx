import { useEffect, useState, useRef } from 'react'
import { Loader2 } from 'lucide-react'
import { getJob } from '../api'

export default function RunProgress({ jobId, totalFiles, onDone }) {
  const [job, setJob] = useState(null)
  const pollRef = useRef(null)

  useEffect(() => {
    pollRef.current = setInterval(async () => {
      try {
        const data = await getJob(jobId)
        setJob(data)
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(pollRef.current)
          onDone(data)
        }
      } catch (e) { /* keep polling */ }
    }, 1200)
    return () => clearInterval(pollRef.current)
  }, [jobId])

  const processed = job?.files_processed_so_far || 0
  const pct = totalFiles > 0 ? Math.min(100, Math.round((processed / totalFiles) * 100)) : 0

  return (
    <div className="flex flex-col items-center justify-center h-full px-8">
      <div className="w-full max-w-md">
        <div className="flex items-center gap-2 justify-center mb-6">
          <Loader2 size={16} className="animate-spin text-[var(--color-accent)]" />
          <span className="text-sm text-[var(--color-text-muted)]">Checking and repairing audio…</span>
        </div>

        {/* waveform-style progress bar */}
        <div className="flex items-end gap-[2px] h-16 mb-4">
          {Array.from({ length: 48 }).map((_, i) => {
            const barPct = (i / 48) * 100
            const active = barPct <= pct
            const height = 20 + Math.abs(Math.sin(i * 0.6)) * 80
            return (
              <div
                key={i}
                className="flex-1 rounded-full transition-colors duration-300"
                style={{
                  height: `${height}%`,
                  backgroundColor: active ? 'var(--color-accent)' : 'var(--color-border)',
                }}
              />
            )
          })}
        </div>

        <div className="flex items-center justify-between text-xs font-mono-data text-[var(--color-text-muted)] mb-6">
          <span>{processed} / {totalFiles || '?'} files</span>
          <span>{pct}%</span>
        </div>

        {job?.log_tail && (
          <div className="bg-[var(--color-panel)] border border-[var(--color-border)] rounded-lg p-3 max-h-40 overflow-y-auto">
            <pre className="text-[10px] font-mono-data text-[var(--color-text-faint)] whitespace-pre-wrap leading-relaxed">
              {job.log_tail}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
