import { Copy, Wrench } from 'lucide-react'

const DEDUP_OPTIONS = [
  { key: 'near', label: 'Exact copies + re-exports', note: 'Recommended. Also catches the same recording resampled, re-leveled or trimmed. Different recordings that merely share tempo or groove are never flagged.' },
  { key: 'exact', label: 'Exact copies only', note: 'Only byte-identical or sample-identical files.' },
  { key: 'off', label: 'Off', note: 'Keep everything.' },
]

export default function RunOptions({ dedupMode, onDedupChange, denoise, onDenoiseChange, extraFormats, onExtraFormatsChange }) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <Copy size={12} className="text-[var(--color-text-muted)]" />
          <span className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wide">Duplicates</span>
        </div>
        <div className="space-y-1.5">
          {DEDUP_OPTIONS.map((o) => (
            <label key={o.key} className={`flex items-start gap-2 p-2 rounded-md border cursor-pointer transition-colors ${dedupMode === o.key ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10' : 'border-[var(--color-border)] hover:border-[var(--color-text-faint)]'}`}>
              <input type="radio" name="dedup" checked={dedupMode === o.key} onChange={() => onDedupChange(o.key)} className="accent-[var(--color-accent)] mt-0.5" />
              <div>
                <div className="text-xs font-medium">{o.label}</div>
                {dedupMode === o.key && <div className="text-[10px] text-[var(--color-text-faint)] leading-snug mt-0.5">{o.note}</div>}
              </div>
            </label>
          ))}
        </div>
        <p className="text-[10px] text-[var(--color-text-faint)] mt-1.5">Duplicates are copied to _duplicates/ — nothing is deleted.</p>
      </div>

      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <Wrench size={12} className="text-[var(--color-text-muted)]" />
          <span className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wide">Repair</span>
        </div>
        <label className="flex items-start gap-2 cursor-pointer py-1">
          <input type="checkbox" checked={!!denoise} onChange={(e) => onDenoiseChange(e.target.checked)} className="accent-[var(--color-accent)] mt-0.5" />
          <div>
            <div className="text-xs">Reduce background noise</div>
            <div className="text-[10px] text-[var(--color-text-faint)] leading-snug">Repairs steady hiss, hum and room noise on files that fail the noise check. Only kept if it measurably improves the file. Best for speech; leave off for music.</div>
          </div>
        </label>
        <label className="flex items-center gap-2 cursor-pointer py-1">
          <input type="checkbox" checked={extraFormats} onChange={(e) => onExtraFormatsChange(e.target.checked)} className="accent-[var(--color-accent)]" />
          <span className="text-xs text-[var(--color-text-muted)]">Also accept mp3 / flac / m4a / ogg</span>
        </label>
      </div>
    </div>
  )
}
