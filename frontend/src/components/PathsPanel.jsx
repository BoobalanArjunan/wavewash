import { FolderOpen, X, CheckCircle2, AlertCircle } from 'lucide-react'

export const PATH_FIELDS = [
  { key: 'input', label: 'Input folder or file', mode: 'dir', required: true,
    placeholder: '/path/to/your/audio  (subfolders are included)',
    help: 'Your originals. This folder is only ever read — never changed.' },
  { key: 'output', label: 'Output folder', mode: 'dir', required: true,
    placeholder: '/path/where/results/should/go',
    help: 'Creates clean/, _error/, _duplicates/ and _logs/ inside it.' },
  { key: 'transcripts', label: 'Transcripts file', mode: 'file', fileExts: ['.csv', '.json'],
    placeholder: 'optional — CSV/JSON: filename,transcript',
    help: 'Enables language and word-count checks against the audio.' },
  { key: 'metadata', label: 'Speaker / session file', mode: 'file', fileExts: ['.csv'],
    placeholder: 'optional — CSV: filename,speaker_id,session_id',
    help: 'Enables speaker-balance and mislabel checks.' },
]

function Status({ field, inputInfo, paths }) {
  if (field.key === 'input' && paths.input) {
    if (inputInfo.error) return <span className="flex items-center gap-1 text-[11px] text-[var(--color-fail)]"><AlertCircle size={11} />{inputInfo.error}</span>
    if (inputInfo.count === 0) return <span className="flex items-center gap-1 text-[11px] text-[var(--color-fail)]"><AlertCircle size={11} />No audio files found here</span>
    if (inputInfo.count > 0) return <span className="flex items-center gap-1 text-[11px] font-mono-data text-[var(--color-pass)]"><CheckCircle2 size={11} />{inputInfo.count} audio file{inputInfo.count !== 1 ? 's' : ''}</span>
  }
  if (field.key === 'output' && paths.output && paths.input && insideOrSame(paths.output, paths.input)) {
    return <span className="flex items-center gap-1 text-[11px] text-[var(--color-fail)]"><AlertCircle size={11} />Can’t be the input folder or inside it</span>
  }
  return null
}

export function insideOrSame(child, parent) {
  const norm = (p) => p.replace(/[\\/]+$/, '')
  const c = norm(child), p = norm(parent)
  return c === p || c.startsWith(p + '/') || c.startsWith(p + '\\')
}

export default function PathsPanel({ paths, onChange, activeField, onActivate, inputInfo, expectedLanguage, onLanguageChange }) {
  return (
    <div className="border-b border-[var(--color-border)] p-4 space-y-3 shrink-0">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">Paths</h2>
        <span className="text-[11px] text-[var(--color-text-faint)]">type / paste, or Browse below</span>
      </div>
      {PATH_FIELDS.map((f) => {
        const active = activeField === f.key
        return (
          <div key={f.key}>
            <div className="flex items-center justify-between mb-1">
              <label htmlFor={`path-${f.key}`} className="text-xs font-medium">
                {f.label}{f.required ? '' : <span className="text-[var(--color-text-faint)] font-normal"> (optional)</span>}
              </label>
              <Status field={f} inputInfo={inputInfo} paths={paths} />
            </div>
            <div className="flex gap-1.5">
              <input
                id={`path-${f.key}`}
                value={paths[f.key]}
                onChange={(e) => onChange(f.key, e.target.value)}
                onFocus={() => onActivate(f.key)}
                spellCheck={false}
                placeholder={f.placeholder}
                className={`flex-1 min-w-0 bg-[var(--color-panel)] border rounded-md px-2.5 py-2 text-xs font-mono-data text-[var(--color-text)] placeholder:text-[var(--color-text-faint)] focus:outline-none ${active ? 'border-[var(--color-accent)]' : 'border-[var(--color-border)]'}`}
              />
              {!f.required && paths[f.key] && (
                <button onClick={() => onChange(f.key, '')} title="Clear"
                  className="px-2 rounded-md border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-fail)]">
                  <X size={13} />
                </button>
              )}
              <button onClick={() => onActivate(f.key)} aria-label={`Browse for ${f.label}`}
                className={`flex items-center gap-1.5 px-2.5 rounded-md border text-xs font-medium transition-colors ${active ? 'bg-[var(--color-accent)] text-[#14161A] border-[var(--color-accent)]' : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-raised)]'}`}>
                <FolderOpen size={13} /> Browse
              </button>
            </div>
            {active && <p className="text-[10px] text-[var(--color-text-faint)] mt-1 leading-snug">{f.help}</p>}
            {f.key === 'transcripts' && paths.transcripts && (
              <div className="flex items-center gap-2 mt-1.5">
                <label htmlFor="expected-lang" className="text-[11px] text-[var(--color-text-muted)]">Expected language</label>
                <input id="expected-lang" value={expectedLanguage} onChange={(e) => onLanguageChange(e.target.value)}
                  placeholder="en" maxLength={5}
                  className="w-16 bg-[var(--color-panel)] border border-[var(--color-border)] rounded px-2 py-1 text-xs font-mono-data focus:outline-none focus:border-[var(--color-accent)]" />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
